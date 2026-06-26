#include "lance_extension.h"

#include "common/enums/storage_format.h"
#include "lance_functions.h"
#include "lance_node_table.h"
#include "lance_rel_table.h"
#include "main/client_context.h"
#include "main/database.h"
#include "storage/storage_manager.h"

namespace lbug {
namespace lance_extension {

using namespace extension;

void LanceExtension::load(main::ClientContext* context) {
    auto& db = *context->getDatabase();
    auto* storageManager = storage::StorageManager::Get(*context);

    // ── Register table factories ────────────────────────────────────────────
    storage::NodeTableFactory nodeFactory =
        [](const storage::StorageManager* sm, const catalog::NodeTableCatalogEntry* entry,
            storage::MemoryManager* mm,
            main::ClientContext* ctx) -> std::unique_ptr<storage::Table> {
        return std::make_unique<LanceNodeTable>(sm, entry, mm, ctx);
    };

    storage::RelTableFactory relFactory =
        [](catalog::RelGroupCatalogEntry* entry, common::table_id_t fromTableID,
            common::table_id_t toTableID, const storage::StorageManager* sm,
            storage::MemoryManager* mm,
            main::ClientContext* ctx) -> std::unique_ptr<storage::Table> {
        return std::make_unique<LanceRelTable>(entry, fromTableID, toTableID, sm, mm, ctx);
    };

    storageManager->registerStorageFormatHandler(common::StorageFormat::LANCE,
        std::move(nodeFactory), std::move(relFactory));

    // ── Register search functions ───────────────────────────────────────────
    ExtensionUtils::addTableFunc<LanceVectorSearchFunction>(db);
    ExtensionUtils::addTableFunc<LanceFTSFunction>(db);
    ExtensionUtils::addTableFunc<LanceHybridSearchFunction>(db);
}

} // namespace lance_extension
} // namespace lbug

#if defined(BUILD_DYNAMIC_LOAD)
extern "C" {
#if defined(_WIN32)
#define INIT_EXPORT __declspec(dllexport)
#else
#define INIT_EXPORT __attribute__((visibility("default")))
#endif

INIT_EXPORT void init(lbug::main::ClientContext* context) {
    lbug::lance_extension::LanceExtension::load(context);
}

INIT_EXPORT const char* name() {
    return lbug::lance_extension::LanceExtension::EXTENSION_NAME;
}
} // extern "C"
#endif
