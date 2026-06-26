#include "lance_node_table.h"

#include <cstring>

#include "catalog/catalog_entry/node_table_catalog_entry.h"
#include "common/arrow/arrow_converter.h"
#include "common/arrow/arrow_nullmask_tree.h"
#include "common/data_chunk/sel_vector.h"
#include "common/exception/runtime.h"
#include "common/file_system/virtual_file_system.h"
#include "common/system_config.h"
#include "common/types/types.h"
#include "lance/lance.hpp"
#include "main/client_context.h"
#include "storage/storage_manager.h"
#include "storage/table/node_table.h"
#include "transaction/transaction.h"

namespace lbug {
namespace lance_extension {

static std::string sanitizeLanceError(std::string message) {
    for (char& ch : message) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    return message;
}

// ─── LanceNodeTableScanSharedState ───────────────────────────────────────────

void LanceNodeTableScanSharedState::reset(ArrowArrayStream newStream) {
    std::lock_guard lock(mtx);
    if (stream_.release)
        stream_.release(&stream_);
    stream_ = newStream;
    std::memset(&newStream, 0, sizeof(newStream)); // disarm the original

    streamExhausted = false;
    currentBatch = nullptr;
    currentBatchGlobalOffset = 0;
    currentMorselStart = 0;

    if (!streamSchemaFetched) {
        // Fetch schema once so each batch's children can be addressed later.
        // The stream schema is the same for all batches.
        if (stream_.get_schema && stream_.get_schema(&stream_, &streamSchema_) == 0) {
            streamSchemaFetched = true;
        }
    }
}

bool LanceNodeTableScanSharedState::readNextBatch() {
    // Caller must hold mtx
    auto newBatch = std::make_shared<LanceBatchData>();
    int rc = stream_.get_next(&stream_, &newBatch->array);
    if (rc != 0 || newBatch->array.release == nullptr) {
        streamExhausted = true;
        return false;
    }
    newBatch->length = static_cast<uint64_t>(newBatch->array.length);

    // Copy the cached stream schema into the batch schema so each batch carries
    // all the structural information ArrowConverter needs.
    if (streamSchemaFetched && streamSchema_.format != nullptr) {
        newBatch->schema = streamSchema_;
        // We do NOT want ~LanceBatchData to release the schema (it's owned by
        // streamSchema_), so clear the release pointer.
        newBatch->schema.release = nullptr;
    }

    size_t prevBatchLength = currentBatch ? currentBatch->length : 0;
    currentBatchGlobalOffset += prevBatchLength;
    currentBatch = std::move(newBatch);
    currentMorselStart = 0;
    return true;
}

LanceNodeTable::~LanceNodeTable() {
    if (schemaCached && cachedSchema_.release) {
        cachedSchema_.release(&cachedSchema_);
    }
}

void LanceNodeTable::ensureDatasetLoaded() const {
    std::lock_guard lock(metadataMtx);
    if (cachedTotalRows != common::INVALID_ROW_IDX && schemaCached) {
        return;
    }

    try {
        auto dataset = lance::Dataset::open(datasetPath);
        if (cachedTotalRows == common::INVALID_ROW_IDX) {
            cachedTotalRows = dataset.count_rows();
        }
        if (!schemaCached) {
            std::memset(&cachedSchema_, 0, sizeof(cachedSchema_));
            dataset.schema(&cachedSchema_);
            schemaCached = true;
            numLanceColumns = static_cast<uint32_t>(cachedSchema_.n_children);
        }
    } catch (const lance::Error& e) {
        throw common::RuntimeException(std::string("Failed to open lance dataset '") + datasetPath +
                                       "': " + sanitizeLanceError(e.what()));
    }
}

bool LanceNodeTableScanSharedState::getNextMorsel(storage::ColumnarNodeTableScanState* scanState) {
    auto* lanceScanState = dynamic_cast<LanceNodeTableScanState*>(scanState);
    if (!lanceScanState)
        return false;

    std::lock_guard lock(mtx);

    while (true) {
        // If there's data remaining in the current batch, carve off a morsel.
        if (currentBatch && currentMorselStart < currentBatch->length) {
            lanceScanState->currentBatch = currentBatch;
            lanceScanState->batchStartGlobalOffset = currentBatchGlobalOffset;
            lanceScanState->morselStart = currentMorselStart;
            lanceScanState->morselEnd =
                std::min(currentMorselStart + morselSize, currentBatch->length);
            currentMorselStart = lanceScanState->morselEnd;
            return true;
        }
        // Need the next batch from the stream.
        if (streamExhausted)
            return false;
        if (!readNextBatch())
            return false;
        // Loop to assign a morsel from the freshly-read batch.
    }
}

// ─── LanceNodeTable ──────────────────────────────────────────────────────────

LanceNodeTable::LanceNodeTable(const storage::StorageManager* storageManager,
    const catalog::NodeTableCatalogEntry* nodeTableEntry, storage::MemoryManager* memoryManager,
    main::ClientContext* context)
    : storage::ColumnarNodeTableBase{storageManager, nodeTableEntry, memoryManager,
          std::make_unique<LanceNodeTableScanSharedState>(kDefaultMorselSize)} {
    // The catalog stores the lance dataset path in the 'storage' field.
    const std::string& storagePath = nodeTableEntry->getStorage();
    if (storagePath.empty()) {
        throw common::RuntimeException(
            "Lance node table has empty storage path. "
            "Specify the dataset path via storage='path/to/dataset.lance'.");
    }
    datasetPath = common::VirtualFileSystem::resolvePath(context, storagePath);
}

void LanceNodeTable::initializeScanCoordination(const transaction::Transaction* transaction) {
    ensureDatasetLoaded();
    auto* lanceSharedState =
        static_cast<LanceNodeTableScanSharedState*>(tableScanSharedState.get());

    try {
        auto dataset = lance::Dataset::open(datasetPath);

        auto scanner = dataset.scan();
        scanner.batch_size(static_cast<int64_t>(kDefaultMorselSize));

        ArrowArrayStream stream;
        std::memset(&stream, 0, sizeof(stream));
        scanner.to_arrow_stream(&stream);

        lanceSharedState->reset(stream);
    } catch (const lance::Error& e) {
        throw common::RuntimeException(
            std::string("Failed to initialize lance scan on '") + datasetPath + "': " + e.what());
    }
}

void LanceNodeTable::initScanState(transaction::Transaction* /*transaction*/,
    storage::TableScanState& scanState, bool /*resetCachedBoundNodeSelVec*/) const {
    auto& lanceScanState = scanState.cast<LanceNodeTableScanState>();

    lanceScanState.initialized = false;
    lanceScanState.scanCompleted = true;

    if (lanceScanState.source == storage::TableScanSource::COMMITTED &&
        lanceScanState.currentBatch != nullptr) {
        lanceScanState.scanCompleted = false;
    }

    lanceScanState.initialized = true;
}

bool LanceNodeTable::scanInternal(transaction::Transaction* /*transaction*/,
    storage::TableScanState& scanState) {
    auto& lanceScanState = scanState.cast<LanceNodeTableScanState>();

    if (lanceScanState.scanCompleted)
        return false;
    if (!lanceScanState.currentBatch || lanceScanState.morselStart >= lanceScanState.morselEnd) {
        lanceScanState.scanCompleted = true;
        return false;
    }

    const auto& batch = *lanceScanState.currentBatch;
    const auto morselStart = lanceScanState.morselStart;
    const auto morselEnd = lanceScanState.morselEnd;
    const auto outputSize = static_cast<uint64_t>(morselEnd - morselStart);
    const auto globalStartOffset = lanceScanState.batchStartGlobalOffset + morselStart;

    scanState.resetOutVectors();
    scanState.outState->getSelVectorUnsafe().setSelSize(outputSize);

    NodeTable::applySemiMaskFilter(scanState, globalStartOffset, outputSize,
        scanState.outState->getSelVectorUnsafe());

    if (scanState.outState->getSelVector().getSelSize() == 0) {
        // Advance offset even if all rows are masked out so we don't loop forever.
        lanceScanState.morselStart += outputSize;
        return false;
    }

    const auto outputToLanceColIdx = getOutputToLanceColumnIdx(scanState.columnIDs);
    copyLanceMorselToOutputVectors(batch, morselStart, outputSize, scanState.outputVectors,
        outputToLanceColIdx);

    const auto tableID = this->getTableID();
    for (uint64_t i = 0; i < outputSize; ++i) {
        auto& nodeID = scanState.nodeIDVector->getValue<common::nodeID_t>(i);
        nodeID.tableID = tableID;
        nodeID.offset = globalStartOffset + i;
    }

    lanceScanState.morselStart += outputSize;
    return true;
}

size_t LanceNodeTable::getNumScanMorsels(const transaction::Transaction* transaction) const {
    auto totalRows = getTotalRowCount(transaction);
    return (totalRows + kDefaultMorselSize - 1) / kDefaultMorselSize;
}

std::unique_ptr<storage::TableScanState> LanceNodeTable::createScanState(
    common::ValueVector* nodeIDVector, const std::vector<common::ValueVector*>& outVectors,
    storage::MemoryManager* memoryManager) const {
    return std::make_unique<LanceNodeTableScanState>(*memoryManager, nodeIDVector, outVectors,
        nodeIDVector->state);
}

bool LanceNodeTable::isVisible(const transaction::Transaction* /*transaction*/,
    common::offset_t offset) const {
    ensureDatasetLoaded();
    return offset < cachedTotalRows;
}

bool LanceNodeTable::isVisibleNoLock(const transaction::Transaction* /*transaction*/,
    common::offset_t offset) const {
    ensureDatasetLoaded();
    return offset < cachedTotalRows;
}

bool LanceNodeTable::lookupPK(const transaction::Transaction* /*transaction*/,
    common::ValueVector* keyVector, uint64_t vectorPos, common::offset_t& result) const {
    if (keyVector->isNull(vectorPos))
        return false;
    ensureDatasetLoaded();

    auto pkColumnID = getPKColumnID();
    int64_t pkLanceIdx = -1;
    for (common::idx_t propIdx = 0; propIdx < nodeTableCatalogEntry->getNumProperties();
         ++propIdx) {
        if (nodeTableCatalogEntry->getColumnID(propIdx) == pkColumnID) {
            pkLanceIdx = static_cast<int64_t>(propIdx);
            break;
        }
    }
    if (pkLanceIdx < 0 || pkLanceIdx >= cachedSchema_.n_children)
        return false;

    auto keyToLookup = keyVector->getAsValue(vectorPos);
    auto pkType = getColumn(pkColumnID).getDataType().copy();
    auto singleState = common::DataChunkState::getSingleValueDataChunkState();
    auto pkVector =
        std::make_unique<common::ValueVector>(std::move(pkType), memoryManager, singleState);
    pkVector->state->setToFlat();

    // Full table scan to find the PK — acceptable for a lookup on an immutable table.
    // For better performance the user should build a lance scalar index on the PK column.
    try {
        auto dataset = lance::Dataset::open(datasetPath);
        auto scanner = dataset.scan();
        scanner.batch_size(4096);
        ArrowArrayStream stream;
        std::memset(&stream, 0, sizeof(stream));
        scanner.to_arrow_stream(&stream);

        ArrowSchema schema;
        std::memset(&schema, 0, sizeof(schema));
        if (stream.get_schema && stream.get_schema(&stream, &schema) != 0) {
            if (stream.release)
                stream.release(&stream);
            return false;
        }

        uint64_t globalOffset = 0;
        ArrowArray batch;
        std::memset(&batch, 0, sizeof(batch));
        while (stream.get_next(&stream, &batch) == 0 && batch.release != nullptr) {
            const auto batchLen = static_cast<uint64_t>(batch.length);
            if (pkLanceIdx < batch.n_children && batch.children[pkLanceIdx] &&
                schema.n_children > pkLanceIdx && schema.children[pkLanceIdx]) {
                auto* childArr = batch.children[pkLanceIdx];
                auto* childSchema = schema.children[pkLanceIdx];
                common::ArrowNullMaskTree nullMask(childSchema, childArr, childArr->offset,
                    childArr->length);
                for (uint64_t rowIdx = 0; rowIdx < batchLen; ++rowIdx) {
                    common::ArrowConverter::fromArrowArray(childSchema, childArr, *pkVector,
                        &nullMask, static_cast<uint64_t>(childArr->offset) + rowIdx, 0, 1);
                    if (!pkVector->isNull(0) && *pkVector->getAsValue(0) == *keyToLookup) {
                        result = globalOffset + rowIdx;
                        if (batch.release)
                            batch.release(&batch);
                        if (schema.release)
                            schema.release(&schema);
                        if (stream.release)
                            stream.release(&stream);
                        return true;
                    }
                }
            }
            if (batch.release)
                batch.release(&batch);
            std::memset(&batch, 0, sizeof(batch));
            globalOffset += batchLen;
        }
        if (schema.release)
            schema.release(&schema);
        if (stream.release)
            stream.release(&stream);
    } catch (const lance::Error& e) {
        throw common::RuntimeException(std::string("Lance PK lookup failed: ") + e.what());
    }
    return false;
}

common::node_group_idx_t LanceNodeTable::getNumBatches(
    const transaction::Transaction* transaction) const {
    auto totalRows = getTotalRowCount(transaction);
    return static_cast<common::node_group_idx_t>(
        (totalRows + kDefaultMorselSize - 1) / kDefaultMorselSize);
}

common::row_idx_t LanceNodeTable::getTotalRowCount(
    const transaction::Transaction* /*transaction*/) const {
    ensureDatasetLoaded();
    return cachedTotalRows;
}

std::vector<int64_t> LanceNodeTable::getOutputToLanceColumnIdx(
    const std::vector<common::column_id_t>& columnIDs) const {
    std::vector<int64_t> result(columnIDs.size(), -1);
    for (size_t col = 0; col < columnIDs.size(); ++col) {
        const auto colID = columnIDs[col];
        if (colID == common::INVALID_COLUMN_ID || colID == common::ROW_IDX_COLUMN_ID)
            continue;
        for (common::idx_t propIdx = 0; propIdx < nodeTableCatalogEntry->getNumProperties();
             ++propIdx) {
            if (nodeTableCatalogEntry->getColumnID(propIdx) == colID) {
                result[col] = static_cast<int64_t>(propIdx);
                break;
            }
        }
    }
    return result;
}

void LanceNodeTable::copyLanceMorselToOutputVectors(const LanceBatchData& batch,
    uint64_t morselStart, uint64_t numRows, const std::vector<common::ValueVector*>& outputVectors,
    const std::vector<int64_t>& outputToLanceColIdx) const {
    if (!batch.array.children || !batch.schema.children)
        return;
    const auto numChildren = static_cast<uint64_t>(batch.array.n_children);

    for (uint64_t outCol = 0; outCol < outputVectors.size(); ++outCol) {
        if (!outputVectors[outCol])
            continue;
        const auto lanceIdx = outputToLanceColIdx[outCol];
        if (lanceIdx < 0 || static_cast<uint64_t>(lanceIdx) >= numChildren)
            continue;
        if (!batch.array.children[lanceIdx] || !batch.schema.children[lanceIdx])
            continue;

        auto* childArray = batch.array.children[lanceIdx];
        auto* childSchema = batch.schema.children[lanceIdx];
        common::ArrowNullMaskTree nullMask(childSchema, childArray, childArray->offset,
            childArray->length);
        common::ArrowConverter::fromArrowArray(childSchema, childArray, *outputVectors[outCol],
            &nullMask, static_cast<uint64_t>(childArray->offset) + morselStart, 0, numRows);
    }
}

} // namespace lance_extension
} // namespace lbug
