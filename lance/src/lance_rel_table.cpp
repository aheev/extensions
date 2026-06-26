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
#include "lance/lance.hpp"
#include "lance_node_table.h"
#include "main/client_context.h"
#include "storage/storage_manager.h"
#include "transaction/transaction.h"

namespace lbug {
namespace lance_extension {

using namespace common;
using namespace storage;
using namespace transaction;

static std::string sanitizeLanceError(std::string message) {
    for (char& ch : message) {
        if (ch == '\n' || ch == '\r') {
            ch = ' ';
        }
    }
    return message;
}

static uint64_t readOffset(const ArrowArray* arr, uint64_t localIdx);

// ─── LanceRelTableScanState ──────────────────────────────────────────────────

LanceRelTableScanState::LanceRelTableScanState(MemoryManager& mm, common::ValueVector* nodeIDVector,
    std::vector<common::ValueVector*> outputVectors,
    std::shared_ptr<common::DataChunkState> outChunkState)
    : RelTableScanState{mm, nodeIDVector, std::move(outputVectors), std::move(outChunkState)} {}

LanceRelTableScanState::~LanceRelTableScanState() = default;

void LanceRelTable::ensureDatasetLoaded() const {
    std::lock_guard lock(metadataMtx);
    if (edgeCacheLoaded && cachedTotalRows != common::INVALID_ROW_IDX) {
        return;
    }

    try {
        std::vector<std::pair<common::offset_t, common::offset_t>> loadedEdges;
        auto dataset = lance::Dataset::open(datasetPath);

        ArrowSchema schema;
        std::memset(&schema, 0, sizeof(schema));
        dataset.schema(&schema);

        auto resolvedFromColumnIdx = fromColumnIdx;
        auto resolvedToColumnIdx = toColumnIdx;
        for (int32_t i = 0; i < schema.n_children; ++i) {
            if (!schema.children[i] || !schema.children[i]->name)
                continue;
            std::string colName = schema.children[i]->name;
            if (colName == "from" || colName == "_from")
                resolvedFromColumnIdx = i;
            else if (colName == "to" || colName == "_to")
                resolvedToColumnIdx = i;
        }

        if (schema.release)
            schema.release(&schema);

        if (resolvedFromColumnIdx < 0 || resolvedToColumnIdx < 0) {
            throw common::RuntimeException("Lance rel table dataset '" + datasetPath +
                                           "' must contain 'from' and 'to' columns. "
                                           "Found fromColumnIdx=" +
                                           std::to_string(resolvedFromColumnIdx) +
                                           " toColumnIdx=" + std::to_string(resolvedToColumnIdx));
        }

        fromColumnIdx = resolvedFromColumnIdx;
        toColumnIdx = resolvedToColumnIdx;

        auto scanner = dataset.scan();
        scanner.batch_size(4096);
        ArrowArrayStream stream;
        std::memset(&stream, 0, sizeof(stream));
        scanner.to_arrow_stream(&stream);

        ArrowArray batch;
        std::memset(&batch, 0, sizeof(batch));
        while (stream.get_next(&stream, &batch) == 0 && batch.release != nullptr) {
            const auto batchLen = static_cast<uint64_t>(batch.length);
            if (batch.children &&
                static_cast<uint64_t>(resolvedFromColumnIdx) <
                    static_cast<uint64_t>(batch.n_children) &&
                static_cast<uint64_t>(resolvedToColumnIdx) <
                    static_cast<uint64_t>(batch.n_children) &&
                batch.children[resolvedFromColumnIdx] && batch.children[resolvedToColumnIdx]) {
                auto* fromArr = batch.children[resolvedFromColumnIdx];
                auto* toArr = batch.children[resolvedToColumnIdx];
                for (uint64_t rowIdx = 0; rowIdx < batchLen; ++rowIdx) {
                    loadedEdges.emplace_back(readOffset(fromArr, rowIdx),
                        readOffset(toArr, rowIdx));
                }
            }
            if (batch.release)
                batch.release(&batch);
            std::memset(&batch, 0, sizeof(batch));
        }
        if (stream.release)
            stream.release(&stream);

        edgeCache = std::move(loadedEdges);
        edgeCacheLoaded = true;
        cachedTotalRows = static_cast<uint64_t>(edgeCache.size());
    } catch (const lance::Error& e) {
        throw common::RuntimeException(std::string("Failed to open lance rel dataset '") +
                                       datasetPath + "': " + sanitizeLanceError(e.what()));
    }
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

// ─── LanceRelTable ───────────────────────────────────────────────────────────

LanceRelTable::LanceRelTable(catalog::RelGroupCatalogEntry* relGroupEntry,
    common::table_id_t fromTableID, common::table_id_t toTableID,
    const StorageManager* storageManager, MemoryManager* memoryManager,
    main::ClientContext* context)
    : ColumnarRelTableBase{relGroupEntry, fromTableID, toTableID, storageManager, memoryManager} {
    const auto& storage = relGroupEntry->getStorage();
    if (storage.empty()) {
        throw common::RuntimeException("Lance rel table has empty storage path. "
                                       "Specify the dataset path via storage='path/to/rel.lance'.");
    }

    datasetPath = common::VirtualFileSystem::resolvePath(context, storage);
}

std::unique_ptr<RelTableScanState> LanceRelTable::createScanState(common::ValueVector* nodeIDVector,
    const std::vector<common::ValueVector*>& outVectors, MemoryManager* memoryManager) const {
    return std::make_unique<LanceRelTableScanState>(*memoryManager, nodeIDVector, outVectors,
        outVectors.empty() ? nodeIDVector->state : outVectors[0]->state);
}

void LanceRelTable::initScanState(Transaction* transaction, TableScanState& scanState,
    bool resetCachedBoundNodeSelVec) const {
    ensureDatasetLoaded();
    auto& relScanState = scanState.cast<RelTableScanState>();
    auto& lanceScanState = static_cast<LanceRelTableScanState&>(relScanState);
    lanceScanState.source = TableScanSource::COMMITTED;
    lanceScanState.nodeGroup = nullptr;
    lanceScanState.nodeGroupIdx = INVALID_NODE_GROUP_IDX;

    if (resetCachedBoundNodeSelVec) {
        if (lanceScanState.nodeIDVector->state->getSelVector().isUnfiltered()) {
            lanceScanState.cachedBoundNodeSelVector.setToUnfiltered();
        } else {
            lanceScanState.cachedBoundNodeSelVector.setToFiltered();
            std::memcpy(lanceScanState.cachedBoundNodeSelVector.getMutableBuffer().data(),
                lanceScanState.nodeIDVector->state->getSelVector().getMutableBuffer().data(),
                lanceScanState.nodeIDVector->state->getSelVector().getSelSize() * sizeof(sel_t));
        }
        lanceScanState.cachedBoundNodeSelVector.setSelSize(
            lanceScanState.nodeIDVector->state->getSelVector().getSelSize());
    }

    // Build the offset → sel-position map (same pattern as ArrowRelTable::initScanState).
    lanceScanState.arrowBoundNodeOffsetToSelPos.clear();
    for (uint64_t i = 0; i < lanceScanState.cachedBoundNodeSelVector.getSelSize(); ++i) {
        const auto boundNodeIdx = lanceScanState.cachedBoundNodeSelVector[i];
        const auto boundNodeID = lanceScanState.nodeIDVector->getValue<nodeID_t>(boundNodeIdx);
        lanceScanState.arrowBoundNodeOffsetToSelPos.emplace(boundNodeID.offset, boundNodeIdx);
    }

    lanceScanState.currentLocalRowIdx = 0;
}

bool LanceRelTable::scanInternal(Transaction* transaction, TableScanState& scanState) {
    auto& lanceScanState =
        static_cast<LanceRelTableScanState&>(scanState.cast<RelTableScanState>());
    return scanFlat(transaction, lanceScanState);
}

// Helper: read a uint64 node offset from an ArrowArray column at a given local row index.
// Lance stores node offsets as int64 or uint64 — both are safe to read as int64 and cast.
static uint64_t readOffset(const ArrowArray* arr, uint64_t localIdx) {
    if (!arr || !arr->buffers || !arr->buffers[1])
        return INVALID_OFFSET;
    const auto* data = static_cast<const int64_t*>(arr->buffers[1]);
    return static_cast<uint64_t>(data[static_cast<size_t>(arr->offset) + localIdx]);
}

bool LanceRelTable::scanFlat(Transaction* /*transaction*/, LanceRelTableScanState& scanState) {
    if (scanState.arrowBoundNodeOffsetToSelPos.empty() || edgeCache.empty()) {
        scanState.outState->getSelVectorUnsafe().setToFiltered(0);
        return false;
    }

    scanState.resetOutVectors();

    const bool isFwd = scanState.direction != RelDataDirection::BWD;
    uint64_t outputCount = 0;
    constexpr uint64_t maxRowsPerCall = DEFAULT_VECTOR_CAPACITY;
    sel_t activeBoundSelPos = INVALID_SEL;
    offset_t activeBoundOffset = INVALID_OFFSET;
    bool hasActiveBound = false;

    while (outputCount < maxRowsPerCall && scanState.currentLocalRowIdx < edgeCache.size()) {
        const auto [fromOffset, toOffset] = edgeCache[scanState.currentLocalRowIdx];
        const auto boundOffset = isFwd ? fromOffset : toOffset;

        const auto it = scanState.arrowBoundNodeOffsetToSelPos.find(boundOffset);
        if (it == scanState.arrowBoundNodeOffsetToSelPos.end()) {
            ++scanState.currentLocalRowIdx;
            continue;
        }

        if (!hasActiveBound) {
            hasActiveBound = true;
            activeBoundOffset = boundOffset;
            activeBoundSelPos = it->second;
        } else if (boundOffset != activeBoundOffset) {
            // Different parent — stop; next scan() call will continue from here.
            break;
        }

        const auto nbrOffset = isFwd ? toOffset : fromOffset;
        const auto nbrTableID = isFwd ? getToNodeTableID() : getFromNodeTableID();
        if (!scanState.outputVectors.empty()) {
            scanState.outputVectors[0]->setValue<internalID_t>(outputCount,
                internalID_t{nbrOffset, nbrTableID});
        }

        for (uint64_t outCol = 1; outCol < scanState.outputVectors.size(); ++outCol) {
            if (outCol >= scanState.columnIDs.size() || !scanState.outputVectors[outCol])
                continue;
            if (scanState.columnIDs[outCol] == REL_ID_COLUMN_ID) {
                scanState.outputVectors[outCol]->setValue<internalID_t>(outputCount,
                    internalID_t{scanState.currentLocalRowIdx, getTableID()});
            }
        }

        ++outputCount;
        ++scanState.currentLocalRowIdx;
    }

    if (outputCount == 0) {
        scanState.outState->getSelVectorUnsafe().setToFiltered(0);
        return false;
    }

    // Point nodeIDVector to the single parent used in this batch (PACKED_EXTEND semantics).
    scanState.setNodeIDVectorToFlat(activeBoundSelPos);
    auto& selVec = scanState.outState->getSelVectorUnsafe();
    selVec.setToFiltered(outputCount);
    for (uint64_t i = 0; i < outputCount; ++i) {
        selVec[i] = static_cast<sel_t>(i);
    }
    return true;
}

common::row_idx_t LanceRelTable::getTotalRowCount(const Transaction* /*transaction*/) const {
    ensureDatasetLoaded();
    return cachedTotalRows;
}

common::row_idx_t LanceRelTable::getActiveBoundNodeCount(const Transaction* /*transaction*/,
    RelDataDirection /*direction*/) const {
    ensureDatasetLoaded();
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
    const Transaction* /*transaction*/, RelDataDirection /*direction*/, common::idx_t /*k*/) const {
    return {};
}

} // namespace lance_extension
} // namespace lbug
