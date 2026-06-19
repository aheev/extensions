#pragma once

#include "function/table/table_function.h"

namespace lbug {
namespace lance_extension {

using function::function_set;

/// LANCE_VECTOR_SEARCH(dataset_path, column, query_vector, k [, metric [, nprobes]])
/// Returns nearest-neighbour rows together with a '_distance' column.
struct LanceVectorSearchFunction {
    static constexpr char name[] = "LANCE_VECTOR_SEARCH";
    static function_set getFunctionSet();
};

/// LANCE_FTS(dataset_path, query [, columns [, max_fuzzy_distance]])
/// Returns full-text search result rows together with a '_score' column.
struct LanceFTSFunction {
    static constexpr char name[] = "LANCE_FTS";
    static function_set getFunctionSet();
};

/// LANCE_HYBRID_SEARCH(dataset_path, column, query_vector, k, fts_query)
/// Returns rows matching both vector and full-text criteria with fusion scoring.
struct LanceHybridSearchFunction {
    static constexpr char name[] = "LANCE_HYBRID_SEARCH";
    static function_set getFunctionSet();
};

} // namespace lance_extension
} // namespace lbug
