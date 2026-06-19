#pragma once

#include <memory>
#include <mutex>
#include <vector>

#include "catalog/catalog_entry/node_table_catalog_entry.h"
#include "common/arrow/arrow.h"
#include "common/exception/runtime.h"
#include "common/types/internal_id_util.h"
#include "storage/table/columnar_node_table_base.h"

namespace lbug {
namespace lance_extension {

/// Owns a single Arrow RecordBatch obtained from an ArrowArrayStream.
/// The ArrowArray is released in the destructor; ArrowSchema is NOT owned
/// (it comes from the shared stream schema and is managed by the shared state).
struct LanceBatchData {
    ArrowSchema schema;
    ArrowArray array;
    uint64_t length = 0;

    LanceBatchData() {
        std::memset(&schema, 0, sizeof(schema));
        std::memset(&array, 0, sizeof(array));
    }
    ~LanceBatchData() {
        if (array.release) array.release(&array);
        // schema.release is intentionally left null (owned by shared state)
    }

    LanceBatchData(const LanceBatchData&) = delete;
    LanceBatchData& operator=(const LanceBatchData&) = delete;
    LanceBatchData(LanceBatchData&&) = delete;
    LanceBatchData& operator=(LanceBatchData&&) = delete;
};

/// Per-thread scan state for LanceNodeTable.
struct LanceNodeTableScanState final : storage::ColumnarNodeTableScanState {
    std::shared_ptr<LanceBatchData> currentBatch;
    uint64_t batchStartGlobalOffset = 0;
    uint64_t morselStart = 0;
    uint64_t morselEnd = 0;

    LanceNodeTableScanState(storage::MemoryManager& mm, common::ValueVector* nodeIDVector,
        std::vector<common::ValueVector*> outputVectors,
        std::shared_ptr<common::DataChunkState> outChunkState)
        : storage::ColumnarNodeTableScanState{mm, nodeIDVector, std::move(outputVectors),
              std::move(outChunkState)} {}
};

/// Shared scan coordination state for LanceNodeTable.
struct LanceNodeTableScanSharedState final : storage::ColumnarNodeTableScanSharedState {
    explicit LanceNodeTableScanSharedState(size_t morselSize) : morselSize(morselSize) {
        std::memset(&stream_, 0, sizeof(stream_));
    }

    ~LanceNodeTableScanSharedState() override {
        if (stream_.release) stream_.release(&stream_);
    }

    void reset(ArrowArrayStream newStream);

    bool getNextMorsel(storage::ColumnarNodeTableScanState* scanState) override;

private:
    bool readNextBatch();

    std::mutex mtx;
    ArrowArrayStream stream_;
    bool streamExhausted = true;

    std::shared_ptr<LanceBatchData> currentBatch;
    uint64_t currentBatchGlobalOffset = 0;
    uint64_t currentMorselStart = 0;
    const size_t morselSize;

    bool streamSchemaFetched = false;
    ArrowSchema streamSchema_;

    LanceNodeTableScanSharedState(const LanceNodeTableScanSharedState&) = delete;
    LanceNodeTableScanSharedState& operator=(const LanceNodeTableScanSharedState&) = delete;
};

/// A node table backed by a Lance dataset.
class LanceNodeTable final : public storage::ColumnarNodeTableBase {
public:
    LanceNodeTable(const storage::StorageManager* storageManager,
        const catalog::NodeTableCatalogEntry* nodeTableEntry,
        storage::MemoryManager* memoryManager, main::ClientContext* context);

    ~LanceNodeTable() override = default;

    void initializeScanCoordination(const transaction::Transaction* transaction) override;

    void initScanState(transaction::Transaction* transaction, storage::TableScanState& scanState,
        bool resetCachedBoundNodeSelVec = true) const override;

    bool scanInternal(transaction::Transaction* transaction,
        storage::TableScanState& scanState) override;

    bool requiresExplicitScanInit() const override { return true; }
    bool usesMorselScan() const override { return true; }
    size_t getNumScanMorsels(const transaction::Transaction* transaction) const override;

    std::unique_ptr<storage::TableScanState> createScanState(common::ValueVector* nodeIDVector,
        const std::vector<common::ValueVector*>& outVectors,
        storage::MemoryManager* memoryManager) const override;

    bool isVisible(const transaction::Transaction* transaction,
        common::offset_t offset) const override;
    bool isVisibleNoLock(const transaction::Transaction* transaction,
        common::offset_t offset) const override;

    bool lookupPK(const transaction::Transaction* transaction, common::ValueVector* keyVector,
        uint64_t vectorPos, common::offset_t& result) const override;

    const std::string& getLanceDatasetPath() const { return datasetPath; }

protected:
    std::string getColumnarFormatName() const override { return "lance"; }
    common::node_group_idx_t getNumBatches(
        const transaction::Transaction* transaction) const override;
    common::row_idx_t getTotalRowCount(const transaction::Transaction* transaction) const override;

private:
    std::vector<int64_t> getOutputToLanceColumnIdx(
        const std::vector<common::column_id_t>& columnIDs) const;

    void copyLanceMorselToOutputVectors(const LanceBatchData& batch, uint64_t morselStart,
        uint64_t numRows, const std::vector<common::ValueVector*>& outputVectors,
        const std::vector<int64_t>& outputToLanceColumnIdx) const;

private:
    std::string datasetPath;
    mutable uint64_t cachedTotalRows = common::INVALID_ROW_IDX;
    uint32_t numLanceColumns = 0;
    ArrowSchema cachedSchema_;
    bool schemaCached = false;

    static constexpr size_t kDefaultMorselSize = 2048;
};

} // namespace lance_extension
} // namespace lbug
