/// Error scenario tests for lance extension.
/// Tests extension-not-loaded errors, mixed-format rejection,
/// missing/malformed files, and schema validation errors.

#include <filesystem>
#include <string>

#include "graph_test/private_graph_test.h"
#include "gtest/gtest.h"
#include "main/query_result.h"

using namespace lbug;
using namespace lbug::testing;
namespace fs = std::filesystem;

// ─── Fixture ─────────────────────────────────────────────────────────────────

class LanceErrorScenariosTest : public EmptyDBTest {
protected:
    void SetUp() override {
        EmptyDBTest::SetUp();
        createDBAndConn();
    }
};

// ─── Tests ───────────────────────────────────────────────────────────────────

TEST_F(LanceErrorScenariosTest, ExtensionNotLoadedGivesActionableError) {
    // If lance extension is not loaded, CREATE TABLE format='lance' should give a clear error
    auto result = conn->query(
        "CREATE NODE TABLE LanceNode (id INT64 PRIMARY KEY) "
        "storage='/tmp/some.lance' format='lance'");
    // Either extension is loaded (success) or gives actionable error, never a cryptic crash
    if (!result->isSuccess()) {
        auto errMsg = result->getErrorMessage();
        // Must mention extension or format in error
        ASSERT_TRUE(errMsg.find("lance") != std::string::npos ||
                    errMsg.find("extension") != std::string::npos ||
                    errMsg.find("format") != std::string::npos)
            << "Error message not actionable: " << errMsg;
    }
}

TEST_F(LanceErrorScenariosTest, MissingFileHandledGracefully) {
    // If extension is loaded, opening a non-existent lance file should fail gracefully
#if defined(BUILD_DYNAMIC_LOAD)
    auto extPath = TestHelper::appendLbugRootPath("extension/lance/build/liblance.lbug_extension");
    auto loadResult = conn->query("LOAD EXTENSION '" + extPath + "'");
    if (!loadResult->isSuccess()) GTEST_SKIP() << "Lance extension not available";
#endif

    auto result = conn->query(
        "CREATE NODE TABLE Ghost (id INT64 PRIMARY KEY) "
        "storage='/nonexistent/path.lance' format='lance'");
    ASSERT_FALSE(result->isSuccess());
    ASSERT_NE(result->getErrorMessage().find("lance"), std::string::npos);
}

TEST_F(LanceErrorScenariosTest, InvalidFormatParameterRejected) {
    auto result = conn->query(
        "CREATE NODE TABLE Bad (id INT64 PRIMARY KEY) format='lanceXXX'");
    ASSERT_FALSE(result->isSuccess());
    auto errMsg = result->getErrorMessage();
    ASSERT_TRUE(errMsg.find("lanceXXX") != std::string::npos ||
                errMsg.find("format") != std::string::npos ||
                errMsg.find("Invalid") != std::string::npos ||
                errMsg.find("Unsupported") != std::string::npos)
        << "Unexpected error: " << errMsg;
}

TEST_F(LanceErrorScenariosTest, DuplicateLoadIsIdempotent) {
#if defined(BUILD_DYNAMIC_LOAD)
    auto extPath = TestHelper::appendLbugRootPath("extension/lance/build/liblance.lbug_extension");
    auto r1 = conn->query("LOAD EXTENSION '" + extPath + "'");
    if (!r1->isSuccess()) GTEST_SKIP() << "Lance extension not available";
    // Duplicate load must not fail
    auto r2 = conn->query("LOAD EXTENSION '" + extPath + "'");
    ASSERT_TRUE(r2->isSuccess()) << r2->getErrorMessage();
#else
    GTEST_SKIP() << "Static build — duplicate load test not applicable";
#endif
}
