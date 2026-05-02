#include "http_config.h"

#include "common/exception/exception.h"
#include "common/types/value/value.h"
#include "function/cast/functions/cast_from_string_functions.h"

namespace lbug {
namespace httpfs_extension {

HTTPConfig::HTTPConfig(main::ClientContext* context) {
    DASSERT(context != nullptr);
    cacheFile =
        context->getCurrentSetting(HTTPCacheFileConfig::HTTP_CACHE_FILE_OPTION).getValue<bool>();
}

void HTTPConfigEnvProvider::setOptionValue(main::ClientContext* context) {
    const auto cacheFileOptionStrVal =
        main::ClientContext::getEnvVariable(HTTPCacheFileConfig::HTTP_CACHE_FILE_ENV_VAR);
    if (cacheFileOptionStrVal != "") {
        bool enableCacheFile = false;
        function::CastString::operation(
            common::string_t{cacheFileOptionStrVal.c_str(), cacheFileOptionStrVal.length()},
            enableCacheFile);
        if (enableCacheFile && context->isInMemory()) {
            throw common::Exception("Cannot enable HTTP file cache when database is in memory");
        }
        context->setExtensionOption(HTTPCacheFileConfig::HTTP_CACHE_FILE_OPTION,
            common::Value::createValue(enableCacheFile));
    }
}

} // namespace httpfs_extension
} // namespace lbug
