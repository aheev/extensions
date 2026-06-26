/// Tests for LanceRelTable.
/// Covers forward/backward scans, filtering, aggregation, and join patterns.

#include <cstring>
#include <filesystem>
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

// ─── Helpers (shared with node table test via inline) ────────────────────────

/// Write a lance dataset with columns [from: int64, to: int64, weight: float]
static void writeLanceRelDataset(const std::string& path, const std::vector<int64_t>& from,
    const std::vector<int64_t>& to, const std::vector<float>& weight = {}) {
    const size_t n = from.size();

    // Build schema
    int numCols = weight.empty() ? 2 : 3;
    ArrowSchema schema;
    std::memset(&schema, 0, sizeof(schema));
    schema.format = "+s";
    schema.name = "";
    schema.n_children = numCols;
    schema.children = new ArrowSchema*[numCols];

    auto makeField = [](const char* name, const char* fmt) {
        auto* s = new ArrowSchema();
        std::memset(s, 0, sizeof(*s));
        s->format = fmt;
        s->name = name;
        return s;
    };
    schema.children[0] = makeField("from", "l"); // int64
    schema.children[1] = makeField("to", "l");   // int64
    if (!weight.empty())
        schema.children[2] = makeField("weight", "f"); // float32
    schema.release = [](ArrowSchema* s) {
        for (int i = 0; i < s->n_children; ++i)
            delete s->children[i];
        delete[] s->children;
        s->release = nullptr;
    };

    auto makeInt64Array = [&](const std::vector<int64_t>& data) {
        auto* buf = new int64_t[data.size()];
        for (size_t i = 0; i < data.size(); ++i)
            buf[i] = data[i];
        auto* a = new ArrowArray();
        std::memset(a, 0, sizeof(*a));
        a->length = data.size();
        a->n_buffers = 2;
        a->buffers = new const void*[2];
        a->buffers[0] = nullptr;
        a->buffers[1] = buf;
        a->release = [](ArrowArray* arr) {
            delete[] static_cast<const int64_t*>(arr->buffers[1]);
            delete[] arr->buffers;
            arr->release = nullptr;
        };
        return a;
    };
    auto makeFloatArray = [&](const std::vector<float>& data) {
        auto* buf = new float[data.size()];
        for (size_t i = 0; i < data.size(); ++i)
            buf[i] = data[i];
        auto* a = new ArrowArray();
        std::memset(a, 0, sizeof(*a));
        a->length = data.size();
        a->n_buffers = 2;
        a->buffers = new const void*[2];
        a->buffers[0] = nullptr;
        a->buffers[1] = buf;
        a->release = [](ArrowArray* arr) {
            delete[] static_cast<const float*>(arr->buffers[1]);
            delete[] arr->buffers;
            arr->release = nullptr;
        };
        return a;
    };

    ArrowArray parent;
    std::memset(&parent, 0, sizeof(parent));
    parent.length = n;
    parent.n_buffers = 1;
    parent.buffers = new const void*[1];
    parent.buffers[0] = nullptr;
    parent.n_children = numCols;
    parent.children = new ArrowArray*[numCols];
    parent.children[0] = makeInt64Array(from);
    parent.children[1] = makeInt64Array(to);
    if (!weight.empty())
        parent.children[2] = makeFloatArray(weight);
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
    ss->array.children = parent.children; // already set

    ArrowArrayStream stream;
    std::memset(&stream, 0, sizeof(stream));
    stream.private_data = ss;
    stream.get_schema = [](ArrowArrayStream* s, ArrowSchema* out) -> int {
        *out = static_cast<SS*>(s->private_data)->schema;
        out->release = nullptr;
        return 0;
    };
    stream.get_next = [](ArrowArrayStream* s, ArrowArray* out) -> int {
        auto* st = static_cast<SS*>(s->private_data);
        if (st->done) {
            std::memset(out, 0, sizeof(*out));
            return 0;
        }
        *out = st->array;
        out->release = nullptr;
        out->children = st->array.children;
        st->done = true;
        return 0;
    };
    stream.release = [](ArrowArrayStream* s) {
        auto* st = static_cast<SS*>(s->private_data);
        if (st->schema.release)
            st->schema.release(&st->schema);
        // Free array children
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

static void writeLanceNodeDatasetInt64(const std::string& path, const std::vector<int64_t>& ids,
    const std::vector<std::string>& names) {
    const size_t n = ids.size();
    // Same as node table helper but using int64 for id
    auto makeField = [](const char* name, const char* fmt) {
        auto* s = new ArrowSchema();
        std::memset(s, 0, sizeof(*s));
        s->format = fmt;
        s->name = name;
        return s;
    };

    ArrowSchema schema;
    std::memset(&schema, 0, sizeof(schema));
    schema.format = "+s";
    schema.n_children = 2;
    schema.children = new ArrowSchema*[2];
    schema.children[0] = makeField("id", "l");
    schema.children[1] = makeField("name", "u");
    schema.release = [](ArrowSchema* s) {
        for (int i = 0; i < s->n_children; ++i)
            delete s->children[i];
        delete[] s->children;
        s->release = nullptr;
    };

    auto* idBuf = new int64_t[n];
    for (size_t i = 0; i < n; ++i)
        idBuf[i] = ids[i];
    auto* idArr = new ArrowArray();
    std::memset(idArr, 0, sizeof(*idArr));
    idArr->length = n;
    idArr->n_buffers = 2;
    idArr->buffers = new const void*[2];
    idArr->buffers[0] = nullptr;
    idArr->buffers[1] = idBuf;
    idArr->release = [](ArrowArray* a) {
        delete[] static_cast<const int64_t*>(a->buffers[1]);
        delete[] a->buffers;
        a->release = nullptr;
    };

    uint32_t totalBytes = 0;
    for (auto& s : names)
        totalBytes += s.size();
    auto* offsets = new int32_t[n + 1];
    auto* values = new char[totalBytes + 1];
    offsets[0] = 0;
    size_t pos = 0;
    for (size_t i = 0; i < n; ++i) {
        std::memcpy(values + pos, names[i].c_str(), names[i].size());
        pos += names[i].size();
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

struct TempDir2 {
    fs::path path;
    TempDir2()
        : path(fs::temp_directory_path() /
               ("lance_rel_" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        fs::create_directories(path);
    }
    ~TempDir2() { fs::remove_all(path); }
    std::string str() const { return path.string(); }
};

// ─── Fixture ─────────────────────────────────────────────────────────────────

class LanceRelTableTest : public EmptyDBTest {
protected:
    void SetUp() override {
        EmptyDBTest::SetUp();
        createDBAndConn();
#if defined(BUILD_DYNAMIC_LOAD)
        auto extPath =
            TestHelper::appendLbugRootPath("extension/lance/build/liblance.lbug_extension");
        auto result = conn->query("LOAD EXTENSION '" + extPath + "'");
        if (!result->isSuccess())
            GTEST_SKIP() << "Lance extension not available: " << result->getErrorMessage();
#endif
    }
};

// ─── Tests ───────────────────────────────────────────────────────────────────

TEST_F(LanceRelTableTest, FwdScanSimple) {
    TempDir2 tmp;
    auto nodePath = (tmp.path / "users.lance").string();
    auto relPath = (tmp.path / "follows.lance").string();
    writeLanceNodeDatasetInt64(nodePath, {0, 1, 2, 3}, {"alice", "bob", "carol", "dan"});
    writeLanceRelDataset(relPath, {0, 0, 1}, {1, 2, 3}, {1.0f, 2.0f, 3.0f});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE User (id INT64 PRIMARY KEY, name STRING) storage='" +
                            nodePath + "' format='lance'")
                    ->isSuccess());
    ASSERT_TRUE(conn->query("CREATE REL TABLE Follows (FROM User TO User, weight FLOAT) storage='" +
                            relPath + "' format='lance'")
                    ->isSuccess());

    auto result = conn->query(
        "MATCH (a:User)-[f:Follows]->(b:User) RETURN a.name, b.name ORDER BY a.name, b.name");
    ASSERT_TRUE(result->isSuccess());
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 3u);
    ASSERT_EQ(rows[0], "alice|bob");
    ASSERT_EQ(rows[1], "alice|carol");
    ASSERT_EQ(rows[2], "bob|dan");
}

TEST_F(LanceRelTableTest, BwdScan) {
    TempDir2 tmp;
    auto nodePath = (tmp.path / "users.lance").string();
    auto relPath = (tmp.path / "follows.lance").string();
    writeLanceNodeDatasetInt64(nodePath, {0, 1, 2}, {"alice", "bob", "carol"});
    writeLanceRelDataset(relPath, {0, 1}, {1, 2});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE U2 (id INT64 PRIMARY KEY, name STRING) storage='" +
                            nodePath + "' format='lance'")
                    ->isSuccess());
    ASSERT_TRUE(
        conn->query("CREATE REL TABLE F2 (FROM U2 TO U2) storage='" + relPath + "' format='lance'")
            ->isSuccess());

    // Backward: who follows carol?
    auto result = conn->query("MATCH (a:U2)<-[:F2]-(b:U2) WHERE a.name = 'carol' RETURN b.name");
    ASSERT_TRUE(result->isSuccess());
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 1u);
    ASSERT_EQ(rows[0], "bob");
}

TEST_F(LanceRelTableTest, RelPropertyQuery) {
    TempDir2 tmp;
    auto nodePath = (tmp.path / "items.lance").string();
    auto relPath = (tmp.path / "edges.lance").string();
    writeLanceNodeDatasetInt64(nodePath, {0, 1, 2}, {"A", "B", "C"});
    writeLanceRelDataset(relPath, {0, 1}, {1, 2}, {10.5f, 20.5f});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE Item (id INT64 PRIMARY KEY, name STRING) storage='" +
                            nodePath + "' format='lance'")
                    ->isSuccess());
    ASSERT_TRUE(conn->query("CREATE REL TABLE Edge (FROM Item TO Item, weight FLOAT) storage='" +
                            relPath + "' format='lance'")
                    ->isSuccess());

    auto result =
        conn->query("MATCH (a:Item)-[e:Edge]->(b:Item) RETURN e.weight ORDER BY e.weight");
    ASSERT_TRUE(result->isSuccess());
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 2u);
}

TEST_F(LanceRelTableTest, RelAggregation) {
    TempDir2 tmp;
    auto nodePath = (tmp.path / "a.lance").string();
    auto relPath = (tmp.path / "b.lance").string();
    writeLanceNodeDatasetInt64(nodePath, {0, 1, 2, 3}, {"x", "y", "z", "w"});
    writeLanceRelDataset(relPath, {0, 0, 1, 1}, {1, 2, 2, 3}, {1.0f, 2.0f, 3.0f, 4.0f});

    ASSERT_TRUE(conn->query("CREATE NODE TABLE N3 (id INT64 PRIMARY KEY, name STRING) storage='" +
                            nodePath + "' format='lance'")
                    ->isSuccess());
    ASSERT_TRUE(conn->query("CREATE REL TABLE R3 (FROM N3 TO N3, w FLOAT) storage='" + relPath +
                            "' format='lance'")
                    ->isSuccess());

    auto result =
        conn->query("MATCH (a:N3)-[r:R3]->(b:N3) RETURN a.name, COUNT(*) ORDER BY a.name");
    ASSERT_TRUE(result->isSuccess());
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 2u);
    // x has 2 outgoing, y has 2 outgoing
    ASSERT_EQ(rows[0], "x|2");
    ASSERT_EQ(rows[1], "y|2");
}
