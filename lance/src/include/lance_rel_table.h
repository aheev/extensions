#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "catalog/catalog_entry/rel_group_catalog_entry.h"
#include "common/arrow/arrow.h"
#include "common/exception/runtime.h"
#include "common/types/internal_id_util.h"
#include "storage/table/columnar_rel_table_base.h"

namespace lbug {
namespace lance_extension {

struct LanceBatchData;

/// Per-thread scan state for LanceRelTable.
struct LanceRelTableScanState final : storage::RelTableScanState {
    /// Index into the LanceRelTable's flat edge cache for resumable morsel scans.
    uint64_t currentLocalRowIdx = 0;

    LanceRelTableScanState(storage::MemoryManager& mm, common::ValueVector* nodeIDVector,
        std::vector<common::ValueVector*> outputVectors,
        std::shared_ptr<common::DataChunkState> outChunkState);

    ~LanceRelTableScanState() override;

    void setToTable(const transaction::Transaction* transaction, storage::Table* table_,
        std::vector<common::column_id_t> columnIDs_,
        std::vector<storage::ColumnPredicateSet> columnPredicateSets_,
        common::RelDataDirection direction_) override;

    LanceRelTableScanState(const LanceRelTableScanState&) = delete;
    LanceRelTableScanState& operator=(const LanceRelTableScanState&) = delete;
};

/// A relationship table backed by a Lance dataset.
class LanceRelTable final : public storage::ColumnarRelTableBase {
public:
    LanceRelTable(catalog::RelGroupCatalogEntry* relGroupEntry, common::table_id_t fromTableID,
        common::table_id_t toTableID, const storage::StorageManager* storageManager,
        storage::MemoryManager* memoryManager, main::ClientContext* context);

    ~LanceRelTable() override = default;

    std::unique_ptr<storage::RelTableScanState> createScanState(common::ValueVector* nodeIDVector,
        const std::vector<common::ValueVector*>& outVectors,
        storage::MemoryManager* memoryManager) const override;

    void initScanState(transaction::Transaction* transaction, storage::TableScanState& scanState,
        bool resetCachedBoundNodeSelVec = true) const override;

    bool scanInternal(transaction::Transaction* transaction,
        storage::TableScanState& scanState) override;

    const std::string& getLanceDatasetPath() const { return datasetPath; }

protected:
    std::string getColumnarFormatName() const override { return "lance"; }
    common::row_idx_t getTotalRowCount(const transaction::Transaction* transaction) const override;
    common::row_idx_t getActiveBoundNodeCount(const transaction::Transaction* transaction,
        common::RelDataDirection direction) const override;
    std::vector<std::pair<common::offset_t, common::row_idx_t>> getAllDegreeEntries(
        const transaction::Transaction* transaction,
        common::RelDataDirection direction) const override;
    std::vector<std::pair<common::offset_t, common::row_idx_t>> getTopKDegreeEntries(
        const transaction::Transaction* transaction, common::RelDataDirection direction,
        common::idx_t k) const override;

private:
    void ensureDatasetLoaded() const;

    bool scanFlat(transaction::Transaction* transaction, LanceRelTableScanState& scanState);

    mutable int32_t fromColumnIdx = -1;
    mutable int32_t toColumnIdx = -1;
    std::string datasetPath;
    mutable uint64_t cachedTotalRows = common::INVALID_ROW_IDX;
    mutable std::vector<std::pair<common::offset_t, common::offset_t>> edgeCache;
    mutable bool edgeCacheLoaded = false;
    mutable std::mutex metadataMtx;
};

} // namespace lance_extension
} // namespace lbug
