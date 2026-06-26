/// Tests for LanceNodeTable.
/// Covers the same scenarios as ArrowNodeTableTest: creation, type conversions,
/// empty tables, large data, and cypher query parity.
///
/// Each test writes a lance dataset to a temp directory, creates a ladybug
/// graph, runs queries, and compares results to expected values.

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "common/arrow/arrow.h"
#include "graph_test/private_graph_test.h"
#include "gtest/gtest.h"
#include "lance/lance.hpp"
#include "main/query_result.h"

using namespace lbug;
using namespace lbug::testing;
namespace fs = std::filesystem;

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Write a simple Int32 + String lance dataset to `path`.
static void writeLanceNodeDataset(const std::string& path, const std::vector<int32_t>& intData,
    const std::vector<std::string>& strData) {
    const size_t n = intData.size();

    // Build Arrow schema: struct<id: int32, name: string>
    ArrowSchema schema;
    std::memset(&schema, 0, sizeof(schema));
    schema.format = "+s";
    schema.n_children = 2;
    schema.children = new ArrowSchema*[2];
    auto* idSchema = new ArrowSchema();
    std::memset(idSchema, 0, sizeof(*idSchema));
    idSchema->format = "i";
    idSchema->name = "id";
    schema.children[0] = idSchema;
    auto* nameSchema = new ArrowSchema();
    std::memset(nameSchema, 0, sizeof(*nameSchema));
    nameSchema->format = "u";
    nameSchema->name = "name";
    schema.children[1] = nameSchema;
    schema.release = [](ArrowSchema* s) {
        for (int i = 0; i < s->n_children; ++i)
            delete s->children[i];
        delete[] s->children;
        s->release = nullptr;
    };

    // Int32 column
    auto* idBuf = new int32_t[n];
    for (size_t i = 0; i < n; ++i)
        idBuf[i] = intData[i];
    auto* idArr = new ArrowArray();
    std::memset(idArr, 0, sizeof(*idArr));
    idArr->length = n;
    idArr->n_buffers = 2;
    idArr->buffers = new const void*[2];
    idArr->buffers[0] = nullptr;
    idArr->buffers[1] = idBuf;
    idArr->release = [](ArrowArray* a) {
        delete[] static_cast<const int32_t*>(a->buffers[1]);
        delete[] a->buffers;
        a->release = nullptr;
    };

    // String column (utf8)
    uint32_t totalBytes = 0;
    for (auto& s : strData)
        totalBytes += s.size();
    auto* offsets = new int32_t[n + 1];
    auto* values = new char[totalBytes + 1];
    offsets[0] = 0;
    size_t pos = 0;
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(values + pos, strData[i].c_str(), strData[i].size());
        pos += strData[i].size();
        offsets[i + 1] = static_cast<int32_t>(pos);
    }
    auto* nameArr = new ArrowArray();
    std::memset(nameArr, 0, sizeof(*nameArr));
    nameArr->length = n;
    nameArr->n_buffers = 3;
    nameArr->buffers = new const void*[3];
    nameArr->buffers[0] = nullptr;
    nameArr->buffers[1] = offsets;
    nameArr->buffers[2] = values;
    nameArr->release = [](ArrowArray* a) {
        delete[] static_cast<const int32_t*>(a->buffers[1]);
        delete[] static_cast<const char*>(a->buffers[2]);
        delete[] a->buffers;
        a->release = nullptr;
    };

    // Struct parent
    ArrowArray parent;
    std::memset(&parent, 0, sizeof(parent));
    parent.length = n;
    parent.n_buffers = 1;
    parent.buffers = new const void*[1];
    parent.buffers[0] = nullptr;
    parent.n_children = 2;
    parent.children = new ArrowArray*[2];
    parent.children[0] = idArr;
    parent.children[1] = nameArr;
    parent.release = [](ArrowArray* a) {
        for (int i = 0; i < a->n_children; ++i) {
            if (a->children[i] && a->children[i]->release)
                a->children[i]->release(a->children[i]);
            delete a->children[i];
        }
        delete[] a->children;
        delete[] a->buffers;
        a->release = nullptr;
    };

    struct SS {
        ArrowSchema schema;
        ArrowArray array;
        bool done = false;
    };
    auto* ss = new SS();
    ss->schema = schema;
    schema.release = nullptr;
    ss->array = parent;
    parent.release = nullptr;
    ss->array.children = parent.children;
    ArrowArrayStream stream;
    std::memset(&stream, 0, sizeof(stream));
    stream.private_data = ss;
    stream.get_schema = [](ArrowArrayStream* s, ArrowSchema* o) -> int {
        *o = static_cast<SS*>(s->private_data)->schema;
        o->release = nullptr;
        return 0;
    };
    stream.get_next = [](ArrowArrayStream* s, ArrowArray* o) -> int {
        auto* st = static_cast<SS*>(s->private_data);
        if (st->done) {
            std::memset(o, 0, sizeof(*o));
            return 0;
        }
        *o = st->array;
        o->release = nullptr;
        o->children = st->array.children;
        st->done = true;
        return 0;
    };
    stream.release = [](ArrowArrayStream* s) {
        auto* st = static_cast<SS*>(s->private_data);
        if (st->schema.release)
            st->schema.release(&st->schema);
        for (int i = 0; i < st->array.n_children; ++i) {
            if (st->array.children[i] && st->array.children[i]->release)
                st->array.children[i]->release(st->array.children[i]);
            delete st->array.children[i];
        }
        delete[] st->array.children;
        delete[] st->array.buffers;
        delete st;
        s->release = nullptr;
    };
    lance::Dataset::write(path, &stream, lance::WriteMode::Create);
}

/// RAII temp directory
struct TempDir {
    fs::path path;
    TempDir()
        : path(fs::temp_directory_path() /
               ("lance_test_" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        fs::create_directories(path);
    }
    ~TempDir() { fs::remove_all(path); }
    std::string str() const { return path.string(); }
};

// ─── Test fixture ─────────────────────────────────────────────────────────────

class LanceNodeTableTest : public EmptyDBTest {
protected:
    void SetUp() override {
        EmptyDBTest::SetUp();
        createDBAndConn();

#if defined(BUILD_DYNAMIC_LOAD)
        auto extPath =
            TestHelper::appendLbugRootPath("extension/lance/build/liblance.lbug_extension");
        auto result = conn->query("LOAD EXTENSION '" + extPath + "'");
        if (!result->isSuccess()) {
            GTEST_SKIP() << "Lance extension not found at " << extPath
                         << " — skipping: " << result->getErrorMessage();
        }
#endif
    }
};

// ─── Tests ───────────────────────────────────────────────────────────────────

TEST_F(LanceNodeTableTest, CreateLanceNodeTableFromVectors) {
    TempDir tmp;
    auto datasetPath = (tmp.path / "nodes.lance").string();
    writeLanceNodeDataset(datasetPath, {1, 2, 3, 4, 5}, {"a", "b", "c", "d", "e"});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE Person (id INT32 PRIMARY KEY, name STRING) "
                            "storage='" +
                            datasetPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query("MATCH (p:Person) RETURN p.id, p.name ORDER BY p.id");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();

    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 5u);
    ASSERT_EQ(rows[0], "1|a");
    ASSERT_EQ(rows[4], "5|e");
}

TEST_F(LanceNodeTableTest, LanceTableCountRows) {
    TempDir tmp;
    auto datasetPath = (tmp.path / "nodes.lance").string();
    std::vector<int32_t> ids;
    std::vector<std::string> names;
    for (int i = 0; i < 100; ++i) {
        ids.push_back(i);
        names.push_back("n" + std::to_string(i));
    }
    writeLanceNodeDataset(datasetPath, ids, names);

    ASSERT_TRUE(conn->query("CREATE NODE TABLE Big (id INT32 PRIMARY KEY, name STRING) "
                            "storage='" +
                            datasetPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query("MATCH (n:Big) RETURN COUNT(*)");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 1u);
    ASSERT_EQ(rows[0], "100");
}

TEST_F(LanceNodeTableTest, LanceTableWithFilter) {
    TempDir tmp;
    auto datasetPath = (tmp.path / "filtered.lance").string();
    writeLanceNodeDataset(datasetPath, {10, 20, 30, 40, 50}, {"x", "y", "z", "w", "v"});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE Item (id INT32 PRIMARY KEY, name STRING) "
                            "storage='" +
                            datasetPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query("MATCH (i:Item) WHERE i.id > 20 RETURN i.id ORDER BY i.id");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 3u);
    ASSERT_EQ(rows[0], "30");
    ASSERT_EQ(rows[2], "50");
}

TEST_F(LanceNodeTableTest, LanceTableAggregation) {
    TempDir tmp;
    auto datasetPath = (tmp.path / "agg.lance").string();
    writeLanceNodeDataset(datasetPath, {1, 2, 3, 4, 5}, {"a", "b", "c", "d", "e"});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE Numbers (id INT32 PRIMARY KEY, name STRING) "
                            "storage='" +
                            datasetPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query("MATCH (n:Numbers) RETURN SUM(n.id)");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 1u);
    ASSERT_EQ(rows[0], "15");
}

TEST_F(LanceNodeTableTest, LanceTableOrderByLimit) {
    TempDir tmp;
    auto datasetPath = (tmp.path / "order.lance").string();
    writeLanceNodeDataset(datasetPath, {5, 3, 1, 4, 2}, {"e", "c", "a", "d", "b"});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE Sorted (id INT32 PRIMARY KEY, name STRING) "
                            "storage='" +
                            datasetPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query("MATCH (s:Sorted) RETURN s.id ORDER BY s.id LIMIT 3");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 3u);
    ASSERT_EQ(rows[0], "1");
    ASSERT_EQ(rows[1], "2");
    ASSERT_EQ(rows[2], "3");
}

TEST_F(LanceNodeTableTest, LanceTableDistinct) {
    TempDir tmp;
    auto datasetPath = (tmp.path / "distinct.lance").string();
    writeLanceNodeDataset(datasetPath, {1, 2, 3, 4, 5}, {"a", "a", "b", "b", "c"});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE D (id INT32 PRIMARY KEY, cat STRING) "
                            "storage='" +
                            datasetPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query("MATCH (d:D) RETURN DISTINCT d.cat ORDER BY d.cat");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 3u);
    ASSERT_EQ(rows[0], "a");
    ASSERT_EQ(rows[1], "b");
    ASSERT_EQ(rows[2], "c");
}

TEST_F(LanceNodeTableTest, LanceTableLargeData) {
    TempDir tmp;
    auto datasetPath = (tmp.path / "large.lance").string();
    const size_t N = 10000;
    std::vector<int32_t> ids(N);
    std::vector<std::string> names(N);
    for (size_t i = 0; i < N; ++i) {
        ids[i] = static_cast<int32_t>(i);
        names[i] = "item_" + std::to_string(i);
    }
    writeLanceNodeDataset(datasetPath, ids, names);

    ASSERT_TRUE(conn->query("CREATE NODE TABLE Large (id INT32 PRIMARY KEY, name STRING) "
                            "storage='" +
                            datasetPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query("MATCH (n:Large) RETURN COUNT(*)");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows[0], std::to_string(N));
}

TEST_F(LanceNodeTableTest, MixedFormatRejected) {
    TempDir tmp;
    auto lanceDataset = (tmp.path / "mixed_nodes.lance").string();
    writeLanceNodeDataset(lanceDataset, {1, 2}, {"a", "b"});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE LNode (id INT32 PRIMARY KEY, name STRING) "
                            "storage='" +
                            lanceDataset + "' format='lance'")
                    ->isSuccess());
    ASSERT_TRUE(conn->query("CREATE NODE TABLE RegNode (id INT64 PRIMARY KEY)")->isSuccess());

    // Attempt to create a lance rel table between lance and regular node tables — should fail
    auto result = conn->query("CREATE REL TABLE BadRel (FROM LNode TO RegNode) "
                              "storage='some.lance' format='lance'");
    ASSERT_FALSE(result->isSuccess());
    ASSERT_NE(result->getErrorMessage().find("Cannot mix lance"), std::string::npos);
}
