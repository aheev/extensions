/// Tests for lance vector search, FTS, and hybrid search functions.

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

struct TempDirVec {
    fs::path path;
    TempDirVec()
        : path(fs::temp_directory_path() /
               ("lance_vec_" +
                   std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()))) {
        fs::create_directories(path);
    }
    ~TempDirVec() { fs::remove_all(path); }
    std::string str() const { return path.string(); }
};

/// Write a lance dataset with a fixed-size vector column [id: int64, vec: fixed_size_list<f32>[4]]
static void writeLanceVectorDataset(const std::string& path, const std::vector<int64_t>& ids,
    const std::vector<std::vector<float>>& vecs) {
    const size_t n = ids.size();
    const int dim = vecs.empty() ? 0 : static_cast<int>(vecs[0].size());

    // Schema
    ArrowSchema schema;
    std::memset(&schema, 0, sizeof(schema));
    schema.format = "+s";
    schema.n_children = 2;
    schema.children = new ArrowSchema*[2];

    auto* idF = new ArrowSchema();
    std::memset(idF, 0, sizeof(*idF));
    idF->format = "l";
    idF->name = "id";
    schema.children[0] = idF;

    char dimFmt[32];
    std::snprintf(dimFmt, sizeof(dimFmt), "+w:%d", dim);
    auto* vecF = new ArrowSchema();
    std::memset(vecF, 0, sizeof(*vecF));
    vecF->format = dimFmt;
    vecF->name = "vec";
    vecF->n_children = 1;
    vecF->children = new ArrowSchema*[1];
    auto* elemF = new ArrowSchema();
    std::memset(elemF, 0, sizeof(*elemF));
    elemF->format = "f";
    elemF->name = "item";
    vecF->children[0] = elemF;
    vecF->release = [](ArrowSchema* s) {
        delete s->children[0];
        delete[] s->children;
        s->release = nullptr;
    };
    schema.children[1] = vecF;
    schema.release = [](ArrowSchema* s) {
        delete s->children[0];
        if (s->children[1]->release)
            s->children[1]->release(s->children[1]);
        delete s->children[1];
        delete[] s->children;
        s->release = nullptr;
    };

    // Id array
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

    // Vec array (fixed_size_list wrapping flat float child)
    auto* elemBuf = new float[n * dim];
    for (size_t i = 0; i < n; ++i)
        for (int j = 0; j < dim; ++j)
            elemBuf[i * dim + j] = vecs[i][j];
    auto* elemArr = new ArrowArray();
    std::memset(elemArr, 0, sizeof(*elemArr));
    elemArr->length = static_cast<int64_t>(n * dim);
    elemArr->n_buffers = 2;
    elemArr->buffers = new const void*[2];
    elemArr->buffers[0] = nullptr;
    elemArr->buffers[1] = elemBuf;
    elemArr->release = [](ArrowArray* a) {
        delete[] static_cast<const float*>(a->buffers[1]);
        delete[] a->buffers;
        a->release = nullptr;
    };

    auto* vecArr = new ArrowArray();
    std::memset(vecArr, 0, sizeof(*vecArr));
    vecArr->length = n;
    vecArr->n_buffers = 1;
    vecArr->buffers = new const void*[1];
    vecArr->buffers[0] = nullptr;
    vecArr->n_children = 1;
    vecArr->children = new ArrowArray*[1];
    vecArr->children[0] = elemArr;
    vecArr->release = [](ArrowArray* a) {
        if (a->children[0] && a->children[0]->release)
            a->children[0]->release(a->children[0]);
        delete a->children[0];
        delete[] a->children;
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
    parent.children[1] = vecArr;
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
        ArrowSchema sc;
        ArrowArray ar;
        bool done = false;
    };
    auto* ss = new SS();
    ss->sc = schema;
    schema.release = nullptr;
    ss->ar = parent;
    parent.release = nullptr;
    ss->ar.children = parent.children;
    ArrowArrayStream stream;
    std::memset(&stream, 0, sizeof(stream));
    stream.private_data = ss;
    stream.get_schema = [](ArrowArrayStream* s, ArrowSchema* o) -> int {
        *o = static_cast<SS*>(s->private_data)->sc;
        o->release = nullptr;
        return 0;
    };
    stream.get_next = [](ArrowArrayStream* s, ArrowArray* o) -> int {
        auto* st = static_cast<SS*>(s->private_data);
        if (st->done) {
            std::memset(o, 0, sizeof(*o));
            return 0;
        }
        *o = st->ar;
        o->release = nullptr;
        o->children = st->ar.children;
        st->done = true;
        return 0;
    };
    stream.release = [](ArrowArrayStream* s) {
        auto* st = static_cast<SS*>(s->private_data);
        if (st->sc.release)
            st->sc.release(&st->sc);
        for (int i = 0; i < st->ar.n_children; ++i) {
            if (st->ar.children[i] && st->ar.children[i]->release)
                st->ar.children[i]->release(st->ar.children[i]);
            delete st->ar.children[i];
        }
        delete[] st->ar.children;
        delete[] st->ar.buffers;
        delete st;
        s->release = nullptr;
    };

    lance::Dataset::write(path, &stream, lance::WriteMode::Create);
}

// ─── Fixture ─────────────────────────────────────────────────────────────────

class LanceVectorSearchTest : public EmptyDBTest {
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

TEST_F(LanceVectorSearchTest, BasicVectorSearch) {
    TempDirVec tmp;
    auto dspath = (tmp.path / "vecs.lance").string();

    writeLanceVectorDataset(dspath, {0, 1, 2, 3},
        {{1.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 1.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f},
            {0.0f, 0.0f, 0.0f, 1.0f}});

    // Query with vector closest to [1, 0, 0, 0] — should return id=0
    auto result = conn->query("CALL LANCE_VECTOR_SEARCH('" + dspath +
                              "', 'vec', [1.0, 0.0, 0.0, 0.0], 1) "
                              "RETURN id");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 1u);
    ASSERT_EQ(rows[0], "0");
}

TEST_F(LanceVectorSearchTest, VectorSearchTopK) {
    TempDirVec tmp;
    auto dspath = (tmp.path / "topk.lance").string();

    writeLanceVectorDataset(dspath, {0, 1, 2},
        {{1.0f, 0.0f, 0.0f, 0.0f}, {0.9f, 0.1f, 0.0f, 0.0f}, {0.0f, 0.0f, 1.0f, 0.0f}});

    auto result = conn->query("CALL LANCE_VECTOR_SEARCH('" + dspath +
                              "', 'vec', [1.0, 0.0, 0.0, 0.0], 2) "
                              "RETURN id ORDER BY id");
    ASSERT_TRUE(result->isSuccess()) << result->getErrorMessage();
    auto rows = TestHelper::convertResultToString(*result);
    ASSERT_EQ(rows.size(), 2u);
    // Top-2 nearest to [1,0,0,0] should be 0 and 1
    ASSERT_NE(std::find(rows.begin(), rows.end(), "0"), rows.end());
    ASSERT_NE(std::find(rows.begin(), rows.end(), "1"), rows.end());
}
