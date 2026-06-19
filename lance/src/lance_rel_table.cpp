#include "lance_rel_table.h"

#include <cstring>

#include "catalog/catalog_entry/rel_group_catalog_entry.h"
#include "common/arrow/arrow_converter.h"
#include "common/arrow/arrow_nullmask_tree.h"
#include "common/data_chunk/sel_vector.h"
#include "common/exception/runtime.h"
#include "common/file_system/virtual_file_system.h"
#include "common/system_config.h"
#include "common/types/internal_id_util.h"
#include "common/types/types.h"
#include "main/client_context.h"
#include "storage/storage_manager.h"
#include "transaction/transaction.h"

#include "lance/lance.hpp"
#include "lance_node_table.h"

namespace lbug {
namespace lance_extension {

using namespace common;
using namespace storage;
using namespace transaction;

// ─── LanceRelTableScanState ──────────────────────────────────────────────────

LanceRelTableScanState::LanceRelTableScanState(MemoryManager& mm,
    common::ValueVector* nodeIDVector, std::vector<common::ValueVector*> outputVectors,
    std::shared_ptr<common::DataChunkState> outChunkState)
    : RelTableScanState{mm, nodeIDVector, std::move(outputVectors), std::move(outChunkState)} {
    std::memset(&stream, 0, sizeof(stream));
    std::memset(&streamSchema, 0, sizeof(streamSchema));
}

LanceRelTableScanState::~LanceRelTableScanState() {
    if (stream.release) stream.release(&stream);
    if (streamSchema.release) streamSchema.release(&streamSchema);
}

void LanceRelTableScanState::setToTable(const Transaction* transaction, Table* table_,
    std::vector<column_id_t> columnIDs_, std::vector<ColumnPredicateSet> columnPredicateSets_,
    RelDataDirection direction_) {
    // Call base class (skips local table setup which lance doesn't support)
    TableScanState::setToTable(transaction, table_, std::move(columnIDs_),
        std::move(columnPredicateSets_));
    columns.resize(columnIDs.size());
    direction = direction_;
    for (size_t i = 0; i < columnIDs.size(); ++i) {
        const auto colID = columnIDs[i];
        if (colID == INVALID_COLUMN_ID || colID == ROW_IDX_COLUMN_ID) {
            columns[i] = nullptr;
        } else {
            columns[i] = table->cast<RelTable>().getColumn(colID, direction);
        }
    }
    csrOffsetColumn = table->cast<RelTable>().getCSROffsetColumn(direction);
    csrLengthColumn = table->cast<RelTable>().getCSRLengthColumn(direction);
    nodeGroupIdx = INVALID_NODE_GROUP_IDX;
}

void LanceRelTableScanState::reset(
    std::unordered_map<common::offset_t, common::sel_t> boundNodeOffsets_) {
    cachedBatchData = nullptr;
    currentBatchStartOffset = 0;
    currentLocalRowIdx = 0;
    boundNodeOffsets = std::move(boundNodeOffsets_);
    // Re-open stream from the rel table
    streamExhausted = false;
    streamInitialized = false;
    schemaFetched = false;
    if (stream.release) stream.release(&stream);
    std::memset(&stream, 0, sizeof(stream));
    if (streamSchema.release) streamSchema.release(&streamSchema);
    std::memset(&streamSchema, 0, sizeof(streamSchema));
}

// ─── LanceRelTable ───────────────────────────────────────────────────────────

LanceRelTable::LanceRelTable(catalog::RelGroupCatalogEntry* relGroupEntry,
    common::table_id_t fromTableID, common::table_id_t toTableID,
    const StorageManager* storageManager, MemoryManager* memoryManager,
    main::ClientContext* context)
    : ColumnarRelTableBase{relGroupEntry, fromTableID, toTableID, storageManager, memoryManager} {
    const auto& storage = relGroupEntry->getStorage();
    if (storage.empty()) {
        throw common::RuntimeException(
            "Lance rel table has empty storage path. "
            "Specify the dataset path via storage='path/to/rel.lance'.");
    }

    datasetPath = common::VirtualFileSystem::resolvePath(context, storage);

    try {
        auto dataset = lance::Dataset::open(datasetPath);
        cachedTotalRows = dataset.count_rows();

        // Discover 'from' and 'to' column indices in the lance schema
        ArrowSchema schema;
        std::memset(&schema, 0, sizeof(schema));
        dataset.schema(&schema);

        for (int32_t i = 0; i < schema.n_children; ++i) {
            if (!schema.children[i] || !schema.children[i]->name) continue;
            std::string colName = schema.children[i]->name;
            if (colName == "from") fromColumnIdx = i;
            else if (colName == "to") toColumnIdx = i;
        }
        if (schema.release) schema.release(&schema);

        if (fromColumnIdx < 0 || toColumnIdx < 0) {
            throw common::RuntimeException(
                "Lance rel table dataset '" + datasetPath +
                "' must contain 'from' and 'to' columns. "
                "Found fromColumnIdx=" + std::to_string(fromColumnIdx) +
                " toColumnIdx=" + std::to_string(toColumnIdx));
        }
    } catch (const lance::Error& e) {
        throw common::RuntimeException(
            std::string("Failed to open lance rel dataset '") + datasetPath + "': " + e.what());
    }
}

void LanceRelTable::initScanState(Transaction* transaction, TableScanState& scanState,
    bool resetCachedBoundNodeSelVec) const {
    auto& relScanState = scanState.cast<RelTableScanState>();
    relScanState.source = TableScanSource::COMMITTED;
    relScanState.nodeGroup = nullptr;
    relScanState.nodeGroupIdx = INVALID_NODE_GROUP_IDX;

    if (resetCachedBoundNodeSelVec) {
        if (relScanState.nodeIDVector->state->getSelVector().isUnfiltered()) {
            relScanState.cachedBoundNodeSelVector.setToUnfiltered();
        } else {
            relScanState.cachedBoundNodeSelVector.setToFiltered();
            std::memcpy(relScanState.cachedBoundNodeSelVector.getMutableBuffer().data(),
                relScanState.nodeIDVector->state->getSelVector().getMutableBuffer().data(),
                relScanState.nodeIDVector->state->getSelVector().getSelSize() * sizeof(sel_t));
        }
        relScanState.cachedBoundNodeSelVector.setSelSize(
            relScanState.nodeIDVector->state->getSelVector().getSelSize());
    }

    auto& lanceScanState = static_cast<LanceRelTableScanState&>(relScanState);

    // Build bound node offsets map
    std::unordered_map<common::offset_t, common::sel_t> boundNodeOffsets;
    for (size_t i = 0; i < lanceScanState.cachedBoundNodeSelVector.getSelSize(); ++i) {
        const sel_t idx = lanceScanState.cachedBoundNodeSelVector[i];
        const auto nodeID = lanceScanState.nodeIDVector->getValue<nodeID_t>(idx);
        boundNodeOffsets.insert({nodeID.offset, idx});
    }
    lanceScanState.reset(std::move(boundNodeOffsets));

    // Open a new lance stream for this scan
    try {
        auto dataset = lance::Dataset::open(datasetPath);
        auto scanner = dataset.scan();
        scanner.batch_size(4096);
        scanner.to_arrow_stream(&lanceScanState.stream);
        lanceScanState.streamInitialized = true;
        lanceScanState.streamExhausted = false;

        if (!lanceScanState.schemaFetched && lanceScanState.stream.get_schema) {
            if (lanceScanState.stream.get_schema(&lanceScanState.stream,
                    &lanceScanState.streamSchema) == 0) {
                lanceScanState.schemaFetched = true;
            }
        }
    } catch (const lance::Error& e) {
        throw common::RuntimeException(
            std::string("Failed to open lance rel scan on '") + datasetPath + "': " + e.what());
    }
}

bool LanceRelTable::scanInternal(Transaction* transaction, TableScanState& scanState) {
    auto& lanceScanState = static_cast<LanceRelTableScanState&>(scanState);
    return scanFlat(transaction, lanceScanState);
}

// Helper: read a uint64 node offset from an ArrowArray column at a given local row index.
// Lance stores node offsets as int64 or uint64 — both are safe to read as int64 and cast.
static uint64_t readOffset(const ArrowArray* arr, uint64_t localIdx) {
    if (!arr || !arr->buffers || !arr->buffers[1]) return INVALID_OFFSET;
    const auto* data = static_cast<const int64_t*>(arr->buffers[1]);
    return static_cast<uint64_t>(data[static_cast<size_t>(arr->offset) + localIdx]);
}

bool LanceRelTable::scanFlat(Transaction* /*transaction*/, LanceRelTableScanState& scanState) {
    scanState.resetOutVectors();

    if (scanState.boundNodeOffsets.empty() || !scanState.streamInitialized ||
        scanState.streamExhausted) {
        scanState.outState->getSelVectorUnsafe().setToFiltered(0);
        return false;
    }

    const bool isFwd = scanState.direction != RelDataDirection::BWD;
    uint64_t totalRowsCollected = 0;
    const uint64_t maxRowsPerCall = DEFAULT_VECTOR_CAPACITY;

    while (totalRowsCollected < maxRowsPerCall) {
        // Load next batch if current one is exhausted
        if (!scanState.cachedBatchData ||
            scanState.currentLocalRowIdx >= scanState.cachedBatchData->length) {
            // Release previous batch
            if (scanState.cachedBatchData) {
                scanState.currentBatchStartOffset += scanState.cachedBatchData->length;
            }
            scanState.currentLocalRowIdx = 0;
            scanState.cachedBatchData = nullptr;

            // Read next batch from stream
            auto newBatch = std::make_shared<LanceBatchData>();
            int rc = scanState.stream.get_next(&scanState.stream, &newBatch->array);
            if (rc != 0 || newBatch->array.release == nullptr) {
                scanState.streamExhausted = true;
                break;
            }
            newBatch->length = static_cast<uint64_t>(newBatch->array.length);
            if (scanState.schemaFetched && scanState.streamSchema.format != nullptr) {
                newBatch->schema = scanState.streamSchema;
                newBatch->schema.release = nullptr; // schema owned by scanState.streamSchema
            }
            scanState.cachedBatchData = std::move(newBatch);
        }

        const auto& batch = *scanState.cachedBatchData;
        if (batch.length == 0 || !batch.array.children || !batch.schema.children) break;

        const auto numChildren = static_cast<uint64_t>(batch.array.n_children);
        if (fromColumnIdx < 0 || toColumnIdx < 0 ||
            static_cast<uint64_t>(fromColumnIdx) >= numChildren ||
            static_cast<uint64_t>(toColumnIdx) >= numChildren) {
            break;
        }

        auto* fromArr = batch.array.children[fromColumnIdx];
        auto* toArr = batch.array.children[toColumnIdx];
        if (!fromArr || !toArr) break;

        for (; scanState.currentLocalRowIdx < batch.length &&
               totalRowsCollected < maxRowsPerCall;
             ++scanState.currentLocalRowIdx) {
            const auto localIdx = scanState.currentLocalRowIdx;

            // Read from/to offsets from the arrow arrays
            // These are stored as uint64 (internal node offsets)
            const auto fromOffset = readOffset(fromArr, localIdx);
            const auto toOffset = readOffset(toArr, localIdx);
            const auto boundOffset = isFwd ? fromOffset : toOffset;

            auto boundIt = scanState.boundNodeOffsets.find(boundOffset);
            if (boundIt == scanState.boundNodeOffsets.end()) continue;

            const auto nbrOffset = isFwd ? toOffset : fromOffset;
            const auto nbrTableID = isFwd ? getToNodeTableID() : getFromNodeTableID();
            const auto globalRowIdx =
                scanState.currentBatchStartOffset + scanState.currentLocalRowIdx;

            // Fill output vectors
            if (!scanState.outputVectors.empty()) {
                scanState.outputVectors[0]->setValue<internalID_t>(
                    totalRowsCollected, internalID_t{nbrOffset, nbrTableID});
            }

            for (uint64_t outCol = 1; outCol < scanState.outputVectors.size(); ++outCol) {
                if (outCol >= scanState.columnIDs.size()) continue;
                const auto colID = scanState.columnIDs[outCol];
                if (colID == INVALID_COLUMN_ID || colID == ROW_IDX_COLUMN_ID ||
                    colID == NBR_ID_COLUMN_ID)
                    continue;
                if (colID == REL_ID_COLUMN_ID) {
                    scanState.outputVectors[outCol]->setValue<internalID_t>(
                        totalRowsCollected, internalID_t{globalRowIdx, getTableID()});
                    continue;
                }
                // Property column: map colID → lance column index
                // Lance columns start after 'from' and 'to' columns.
                // Property index = colID - 2 (assuming columns are ordered after from/to)
                // A more robust approach would use the schema names, but we use colID directly.
                const int64_t lanceColIdx =
                    static_cast<int64_t>(colID) + 2; // +2 to skip from, to
                if (lanceColIdx < 0 || static_cast<uint64_t>(lanceColIdx) >= numChildren)
                    continue;
                auto* propArr = batch.array.children[lanceColIdx];
                auto* propSchema = batch.schema.children[lanceColIdx];
                if (!propArr || !propSchema) continue;

                common::ArrowNullMaskTree nullMask(propSchema, propArr, propArr->offset, propArr->length);
                common::ArrowConverter::fromArrowArray(propSchema, propArr,
                    *scanState.outputVectors[outCol], &nullMask,
                    static_cast<uint64_t>(propArr->offset) + localIdx, totalRowsCollected, 1);
            }

            // Assign the bound node ID for join-back
            if (scanState.nodeIDVector) {
                scanState.nodeIDVector->setValue<internalID_t>(
                    totalRowsCollected,
                    internalID_t{boundOffset,
                        isFwd ? getFromNodeTableID() : getToNodeTableID()});
            }

            ++totalRowsCollected;
        }

        // If we read the entire batch without filling the output buffer, continue to next batch.
        if (scanState.currentLocalRowIdx >= batch.length) continue;
        // Otherwise, we filled the buffer; return what we have.
        break;
    }

    if (totalRowsCollected == 0) {
        scanState.outState->getSelVectorUnsafe().setToFiltered(0);
        return false;
    }
    scanState.outState->getSelVectorUnsafe().setSelSize(totalRowsCollected);
    return true;
}

common::row_idx_t LanceRelTable::getTotalRowCount(
    const Transaction* /*transaction*/) const {
    if (cachedTotalRows != INVALID_ROW_IDX) return cachedTotalRows;
    try {
        auto dataset = lance::Dataset::open(datasetPath);
        cachedTotalRows = dataset.count_rows();
    } catch (const lance::Error& e) {
        throw common::RuntimeException(
            std::string("Failed to count rows in lance rel dataset '") + datasetPath + "': " +
            e.what());
    }
    return cachedTotalRows;
}

common::row_idx_t LanceRelTable::getActiveBoundNodeCount(
    const Transaction* /*transaction*/, RelDataDirection /*direction*/) const {
    // Return estimate: assume each bound node has at least one relationship
    return cachedTotalRows != INVALID_ROW_IDX ? cachedTotalRows : 0;
}

std::vector<std::pair<common::offset_t, common::row_idx_t>> LanceRelTable::getAllDegreeEntries(
    const Transaction* /*transaction*/, RelDataDirection /*direction*/) const {
    // Full degree computation for lance tables: would require scanning 'from'/'to' columns
    // and counting occurrences. This is only called for stats; return empty for now.
    return {};
}

std::vector<std::pair<common::offset_t, common::row_idx_t>> LanceRelTable::getTopKDegreeEntries(
    const Transaction* /*transaction*/, RelDataDirection /*direction*/,
    common::idx_t /*k*/) const {
    return {};
}

} // namespace lance_extension
} // namespace lbug
