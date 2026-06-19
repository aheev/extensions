#include "lance_functions.h"

#include <cstring>
#include <memory>
#include <vector>

#include "binder/binder.h"
#include "common/arrow/arrow.h"
#include "common/arrow/arrow_converter.h"
#include "common/arrow/arrow_nullmask_tree.h"
#include "common/exception/runtime.h"
#include "common/file_system/virtual_file_system.h"
#include "common/types/value/nested.h"
#include "common/types/value/value.h"
#include "function/table/bind_data.h"
#include "function/table/bind_input.h"
#include "function/table/table_function.h"
#include "main/client_context.h"
#include "processor/execution_context.h"

#include "lance/lance.hpp"

namespace lbug {
namespace lance_extension {

using namespace function;
using namespace common;

// ─── Vector Search ───────────────────────────────────────────────────────────

struct LanceVectorSearchBindData : TableFuncBindData {
    std::string datasetPath;
    std::string columnName;
    std::vector<float> queryVector;
    uint32_t k;
    std::string metric; // "cosine", "l2", "dot"
    uint32_t nprobes;

    LanceVectorSearchBindData(std::string path, std::string col, std::vector<float> query,
        uint32_t k, std::string metric, uint32_t nprobes,
        binder::expression_vector columns)
        : TableFuncBindData{std::move(columns)}, datasetPath{std::move(path)},
          columnName{std::move(col)}, queryVector{std::move(query)}, k{k},
          metric{std::move(metric)}, nprobes{nprobes} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<LanceVectorSearchBindData>(datasetPath, columnName, queryVector, k,
            metric, nprobes, columns);
    }
};

struct LanceSearchSharedState : TableFuncSharedState {
    ArrowArrayStream stream;
    bool exhausted = false;
    std::mutex streamMtx;

    LanceSearchSharedState() { std::memset(&stream, 0, sizeof(stream)); }

    ~LanceSearchSharedState() override {
        if (stream.release) stream.release(&stream);
    }
};

static std::unique_ptr<TableFuncBindData> bindVectorSearch(main::ClientContext* context,
    const TableFuncBindInput* input) {
    if (input->params.size() < 4) {
        throw RuntimeException("LANCE_VECTOR_SEARCH requires at least 4 arguments: "
                               "dataset_path, column, query_vector, k");
    }
    auto datasetPath = input->getLiteralVal<std::string>(0);
    auto columnName = input->getLiteralVal<std::string>(1);
    // query_vector is a LIST of floats (passed as a literal Value with children)
    auto vecValue = input->getValue(2);
    std::vector<float> queryVec;
    for (uint32_t i = 0; i < vecValue.getChildrenSize(); ++i) {
        queryVec.push_back(
            static_cast<float>(NestedVal::getChildVal(&vecValue, i)->getValue<double>()));
    }
    auto k = static_cast<uint32_t>(input->getLiteralVal<int64_t>(3));
    std::string metric = (input->params.size() > 4) ? input->getLiteralVal<std::string>(4) : "l2";
    uint32_t nprobes = (input->params.size() > 5)
                           ? static_cast<uint32_t>(input->getLiteralVal<int64_t>(5))
                           : 1;

    // Open dataset to discover schema
    auto resolvedPath = VirtualFileSystem::resolvePath(context, datasetPath);
    try {
        auto dataset = lance::Dataset::open(resolvedPath);
        ArrowSchema schema;
        std::memset(&schema, 0, sizeof(schema));
        dataset.schema(&schema);

        std::vector<LogicalType> returnTypes;
        std::vector<std::string> returnNames;
        for (int32_t i = 0; i < schema.n_children; ++i) {
            if (!schema.children[i] || !schema.children[i]->name) continue;
            returnNames.push_back(schema.children[i]->name);
            returnTypes.push_back(ArrowConverter::fromArrowSchema(schema.children[i]));
        }
        // Add the _distance column
        returnNames.push_back("_distance");
        returnTypes.push_back(LogicalType{LogicalTypeID::FLOAT});

        if (schema.release) schema.release(&schema);

        auto columns = input->binder->createVariables(returnNames, returnTypes);
        return std::make_unique<LanceVectorSearchBindData>(std::move(resolvedPath),
            std::move(columnName), std::move(queryVec), k, std::move(metric), nprobes,
            std::move(columns));
    } catch (const lance::Error& e) {
        throw RuntimeException(std::string("LANCE_VECTOR_SEARCH bind failed: ") + e.what());
    }
}

static std::shared_ptr<TableFuncSharedState> initVectorSearchSharedState(
    const TableFuncInitSharedStateInput& input) {
    auto* bindData = input.bindData->constPtrCast<LanceVectorSearchBindData>();
    auto state = std::make_shared<LanceSearchSharedState>();

    try {
        auto dataset = lance::Dataset::open(bindData->datasetPath);
        auto scanner = dataset.scan();

        LanceMetricType metric = LANCE_METRIC_L2;
        if (bindData->metric == "cosine") metric = LANCE_METRIC_COSINE;
        else if (bindData->metric == "dot")   metric = LANCE_METRIC_DOT;

        scanner.nearest(bindData->columnName, bindData->queryVector.data(),
                        bindData->queryVector.size(), bindData->k)
            .nprobes(bindData->nprobes)
            .metric(metric);

        scanner.to_arrow_stream(&state->stream);
    } catch (const lance::Error& e) {
        throw RuntimeException(std::string("LANCE_VECTOR_SEARCH init failed: ") + e.what());
    }

    return state;
}

static offset_t vectorSearchTableFunc(const TableFuncInput& input, TableFuncOutput& output) {
    auto* sharedState = input.sharedState->ptrCast<LanceSearchSharedState>();
    if (sharedState->exhausted) return 0;

    ArrowArray batch;
    std::memset(&batch, 0, sizeof(batch));
    {
        std::lock_guard lock{sharedState->streamMtx};
        int rc = sharedState->stream.get_next(&sharedState->stream, &batch);
        if (rc != 0 || batch.release == nullptr) {
            sharedState->exhausted = true;
            return 0;
        }
    }

    auto batchLen = static_cast<uint64_t>(batch.length);
    ArrowSchema schema;
    std::memset(&schema, 0, sizeof(schema));
    sharedState->stream.get_schema(&sharedState->stream, &schema);

    // Copy each column from the batch to the output data chunk
    for (uint64_t col = 0; col < static_cast<uint64_t>(output.dataChunk.getNumValueVectors()) &&
                            col < static_cast<uint64_t>(batch.n_children) &&
                            col < static_cast<uint64_t>(schema.n_children);
         ++col) {
        auto* childArr = batch.children[col];
        auto* childSchema = schema.children[col];
        if (!childArr || !childSchema) continue;
        ArrowNullMaskTree nullMask(childSchema, childArr, childArr->offset, childArr->length);
        ArrowConverter::fromArrowArray(childSchema, childArr, output.dataChunk.getValueVectorMutable(col),
            &nullMask, static_cast<uint64_t>(childArr->offset), 0, batchLen);
    }

    if (schema.release) schema.release(&schema);
    if (batch.release) batch.release(&batch);
    output.dataChunk.state->getSelVectorUnsafe().setSelSize(batchLen);
    return batchLen;
}

function_set LanceVectorSearchFunction::getFunctionSet() {
    function_set functionSet;
    auto func = std::make_unique<TableFunction>(name,
        std::vector<LogicalTypeID>{LogicalTypeID::STRING, LogicalTypeID::STRING,
            LogicalTypeID::LIST, LogicalTypeID::INT64});
    func->bindFunc = bindVectorSearch;
    func->initSharedStateFunc = initVectorSearchSharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->tableFunc = vectorSearchTableFunc;
    func->canParallelFunc = [] { return false; }; // stream is sequential
    functionSet.push_back(std::move(func));
    return functionSet;
}

// ─── FTS ─────────────────────────────────────────────────────────────────────

struct LanceFTSBindData : TableFuncBindData {
    std::string datasetPath;
    std::string query;
    std::vector<std::string> searchColumns; // renamed from 'columns' to avoid base class clash
    uint32_t maxFuzzyDistance;

    LanceFTSBindData(std::string path, std::string query, std::vector<std::string> cols,
        uint32_t maxFuzzy, binder::expression_vector outputColumns)
        : TableFuncBindData{std::move(outputColumns)}, datasetPath{std::move(path)},
          query{std::move(query)}, searchColumns{std::move(cols)}, maxFuzzyDistance{maxFuzzy} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<LanceFTSBindData>(datasetPath, query, searchColumns,
            maxFuzzyDistance, columns);
    }
};

static std::unique_ptr<TableFuncBindData> bindFTS(main::ClientContext* context,
    const TableFuncBindInput* input) {
    if (input->params.size() < 2) {
        throw RuntimeException("LANCE_FTS requires at least 2 arguments: dataset_path, query");
    }
    auto datasetPath = input->getLiteralVal<std::string>(0);
    auto query = input->getLiteralVal<std::string>(1);
    std::vector<std::string> searchCols;
    if (input->params.size() > 2) {
        searchCols.push_back(input->getLiteralVal<std::string>(2));
    }
    uint32_t maxFuzzy = (input->params.size() > 3)
                            ? static_cast<uint32_t>(input->getLiteralVal<int64_t>(3))
                            : 0;

    auto resolvedPath = VirtualFileSystem::resolvePath(context, datasetPath);
    try {
        auto dataset = lance::Dataset::open(resolvedPath);
        ArrowSchema schema;
        std::memset(&schema, 0, sizeof(schema));
        dataset.schema(&schema);

        std::vector<LogicalType> returnTypes;
        std::vector<std::string> returnNames;
        for (int32_t i = 0; i < schema.n_children; ++i) {
            if (!schema.children[i] || !schema.children[i]->name) continue;
            returnNames.push_back(schema.children[i]->name);
            returnTypes.push_back(ArrowConverter::fromArrowSchema(schema.children[i]));
        }
        returnNames.push_back("_score");
        returnTypes.push_back(LogicalType{LogicalTypeID::FLOAT});

        if (schema.release) schema.release(&schema);
        auto columns = input->binder->createVariables(returnNames, returnTypes);
        return std::make_unique<LanceFTSBindData>(std::move(resolvedPath), std::move(query),
            std::move(searchCols), maxFuzzy, std::move(columns));
    } catch (const lance::Error& e) {
        throw RuntimeException(std::string("LANCE_FTS bind failed: ") + e.what());
    }
}

static std::shared_ptr<TableFuncSharedState> initFTSSharedState(
    const TableFuncInitSharedStateInput& input) {
    auto* bindData = input.bindData->constPtrCast<LanceFTSBindData>();
    auto state = std::make_shared<LanceSearchSharedState>();
    try {
        auto dataset = lance::Dataset::open(bindData->datasetPath);
        dataset.scan()
            .full_text_search(bindData->query, bindData->searchColumns, bindData->maxFuzzyDistance)
            .to_arrow_stream(&state->stream);
    } catch (const lance::Error& e) {
        throw RuntimeException(std::string("LANCE_FTS init failed: ") + e.what());
    }
    return state;
}

function_set LanceFTSFunction::getFunctionSet() {
    function_set functionSet;
    auto func = std::make_unique<TableFunction>(
        name, std::vector<LogicalTypeID>{LogicalTypeID::STRING, LogicalTypeID::STRING});
    func->bindFunc = bindFTS;
    func->initSharedStateFunc = initFTSSharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->tableFunc = vectorSearchTableFunc; // reuse same Arrow stream output logic
    func->canParallelFunc = [] { return false; };
    functionSet.push_back(std::move(func));
    return functionSet;
}

// ─── Hybrid Search ───────────────────────────────────────────────────────────

struct LanceHybridSearchBindData : TableFuncBindData {
    std::string datasetPath;
    std::string vectorColumn;
    std::vector<float> queryVector;
    uint32_t k;
    std::string ftsQuery;

    LanceHybridSearchBindData(std::string path, std::string col, std::vector<float> query,
        uint32_t k, std::string ftsQuery, binder::expression_vector columns)
        : TableFuncBindData{std::move(columns)}, datasetPath{std::move(path)},
          vectorColumn{std::move(col)}, queryVector{std::move(query)}, k{k},
          ftsQuery{std::move(ftsQuery)} {}

    std::unique_ptr<TableFuncBindData> copy() const override {
        return std::make_unique<LanceHybridSearchBindData>(datasetPath, vectorColumn, queryVector,
            k, ftsQuery, columns);
    }
};

static std::unique_ptr<TableFuncBindData> bindHybridSearch(main::ClientContext* context,
    const TableFuncBindInput* input) {
    if (input->params.size() < 5) {
        throw RuntimeException(
            "LANCE_HYBRID_SEARCH requires 5 arguments: dataset_path, column, query_vector, k, fts_query");
    }
    auto datasetPath = input->getLiteralVal<std::string>(0);
    auto columnName = input->getLiteralVal<std::string>(1);
    auto vecValue = input->getValue(2);
    std::vector<float> queryVec;
    for (uint32_t i = 0; i < vecValue.getChildrenSize(); ++i) {
        queryVec.push_back(
            static_cast<float>(NestedVal::getChildVal(&vecValue, i)->getValue<double>()));
    }
    auto k = static_cast<uint32_t>(input->getLiteralVal<int64_t>(3));
    auto ftsQuery = input->getLiteralVal<std::string>(4);

    auto resolvedPath = VirtualFileSystem::resolvePath(context, datasetPath);
    try {
        auto dataset = lance::Dataset::open(resolvedPath);
        ArrowSchema schema;
        std::memset(&schema, 0, sizeof(schema));
        dataset.schema(&schema);

        std::vector<LogicalType> returnTypes;
        std::vector<std::string> returnNames;
        for (int32_t i = 0; i < schema.n_children; ++i) {
            if (!schema.children[i] || !schema.children[i]->name) continue;
            returnNames.push_back(schema.children[i]->name);
            returnTypes.push_back(ArrowConverter::fromArrowSchema(schema.children[i]));
        }
        returnNames.push_back("_score");
        returnTypes.push_back(LogicalType{LogicalTypeID::FLOAT});

        if (schema.release) schema.release(&schema);
        auto columns = input->binder->createVariables(returnNames, returnTypes);
        return std::make_unique<LanceHybridSearchBindData>(std::move(resolvedPath),
            std::move(columnName), std::move(queryVec), k, std::move(ftsQuery),
            std::move(columns));
    } catch (const lance::Error& e) {
        throw RuntimeException(std::string("LANCE_HYBRID_SEARCH bind failed: ") + e.what());
    }
}

static std::shared_ptr<TableFuncSharedState> initHybridSearchSharedState(
    const TableFuncInitSharedStateInput& input) {
    auto* bindData = input.bindData->constPtrCast<LanceHybridSearchBindData>();
    auto state = std::make_shared<LanceSearchSharedState>();
    try {
        auto dataset = lance::Dataset::open(bindData->datasetPath);
        // Run vector search + FTS; the scanner merges results internally
        dataset.scan()
            .nearest(bindData->vectorColumn, bindData->queryVector.data(),
                bindData->queryVector.size(), bindData->k)
            .full_text_search(bindData->ftsQuery)
            .to_arrow_stream(&state->stream);
    } catch (const lance::Error& e) {
        throw RuntimeException(std::string("LANCE_HYBRID_SEARCH init failed: ") + e.what());
    }
    return state;
}

function_set LanceHybridSearchFunction::getFunctionSet() {
    function_set functionSet;
    auto func = std::make_unique<TableFunction>(name,
        std::vector<LogicalTypeID>{LogicalTypeID::STRING, LogicalTypeID::STRING,
            LogicalTypeID::LIST, LogicalTypeID::INT64, LogicalTypeID::STRING});
    func->bindFunc = bindHybridSearch;
    func->initSharedStateFunc = initHybridSearchSharedState;
    func->initLocalStateFunc = TableFunction::initEmptyLocalState;
    func->tableFunc = vectorSearchTableFunc; // reuse Arrow stream output logic
    func->canParallelFunc = [] { return false; };
    functionSet.push_back(std::move(func));
    return functionSet;
}

} // namespace lance_extension
} // namespace lbug
