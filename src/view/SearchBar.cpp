// view/SearchBar.cpp — the Ctrl+F fuzzy-search overlay (spec §8.4).
//
// A no-title, always-on-top window near top-center. The query is fed to
// SearchIndex::query (a linear fuzzy scan, <5ms even at ~1M entries — spec §7.4)
// so typing stays latency-friendly. Selecting a hit resolves it: expand any
// collapsed ancestor group, set the selection, and fly the camera to the node's
// layout box center (spec §8.4).
//
// THREADING: main-thread only. Reads the already-built SearchIndex + published
// LayoutResult; never touches worker state.
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "imgui.h"

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h that
// App.h pulls in without including it; include it first so App.h compiles.
#include "engine/LayoutEngine.h"
#include "engine/SearchIndex.h"
#include "ir/IR.h"
#include "view/App.h"
#include "view/PanelHelpers.h"

namespace netvis {

namespace {

// #123: the result set for a query, cached across frames.
//
// Both the overlay and the results panel used to call index.query() EVERY frame
// — two full linear scans of the whole index per frame when both are open. The
// original comment called that "cheap enough per spec §7.4 … without extra
// caching state", which held when the index was small; at ~1M entries it is two
// complete sweeps per frame for a result set that only changes when the query,
// the model, or the index does.
//
// The key must include the model POINTER and the index identity, not just the
// query string. Per-tab JobSystems mean generation counters all start at 0, so
// two tabs can present an identical (generation, graph) pair — this project has
// shipped that bug three times. Entry pointer + count identifies the index a
// hit's `entry` field indexes into, which is the thing that must not go stale:
// a hit resolved against a rebuilt index would read the wrong entry.
//
// `limit` is part of the key rather than serving a smaller request from a larger
// cached one. query() applies the cap during ranking, so the first 100 of a
// 500-hit request are not guaranteed to be the same 100 a limit-100 request
// returns, and quietly assuming otherwise would silently change what the overlay
// shows.
struct QueryCache {
  const SearchEntry* entries = nullptr;
  size_t entry_count = 0;
  const ir::Model* model = nullptr;
  std::string query;
  bool types_ready = false;
  uint32_t limit = 0;
  bool valid = false;
  std::vector<SearchHit> hits;
};

const std::vector<SearchHit>& cached_query(const SearchIndex& index,
                                           const ir::Model& model,
                                           const std::string& query,
                                           bool types_ready, uint32_t limit,
                                           QueryCache& cache) {
  const SearchEntry* entries =
      index.entries().empty() ? nullptr : index.entries().data();
  const bool hit = cache.valid && cache.entries == entries &&
                   cache.entry_count == index.entries().size() &&
                   cache.model == &model && cache.types_ready == types_ready &&
                   cache.limit == limit && cache.query == query;
  if (hit) return cache.hits;

  cache.entries = entries;
  cache.entry_count = index.entries().size();
  cache.model = &model;
  cache.query = query;
  cache.types_ready = types_ready;
  cache.limit = limit;
  cache.hits = index.query(query, model, types_ready, limit);
  cache.valid = true;
  return cache.hits;
}

// Short tag for the results list, per hit kind.
const char* kind_tag(SearchKind k) {
  switch (k) {
    case SearchKind::Node: return "node";
    case SearchKind::OpType: return "op";
    case SearchKind::Value: return "value";
    case SearchKind::Tensor: return "tensor";
  }
  return "?";
}

// Resolve a hit's producer node index within its graph, or -1. For Node hits
// that's ref directly; for Value hits it's the value's producer.
int32_t node_index_for_hit(const ir::Model& model, const SearchEntry& e) {
  if (e.graph >= model.graphs.size()) return -1;
  const ir::Graph& g = model.graphs[e.graph];
  switch (e.kind) {
    case SearchKind::Node:
      return e.ref < g.nodes.size() ? static_cast<int32_t>(e.ref) : -1;
    case SearchKind::Value:
      if (e.ref < g.values.size()) return g.values[e.ref].producer;
      return -1;
    default:
      return -1;
  }
}

// Commit navigation to a chosen result (Enter / click, spec §8.4).
void resolve_hit(App& app, const SearchEntry& e) {
  ModelSession& session = app.session();
  const ir::Model* model = session.model();
  if (!model) return;

  // A flat tensor (has_graph==false) has no node to fly to — inspect it instead.
  if (e.kind == SearchKind::Tensor) {
    if (e.ref < model->flat_tensors.size()) {
      app.inspect_tensor(model->flat_tensors[e.ref]);
      app.view().table_selected_row = static_cast<int64_t>(e.ref);
    }
    app.view().search_open = false;
    return;
  }

  // Ensure we are viewing the graph the hit lives in (subgraph dives, spec §8.4).
  if (e.graph != session.current_graph() && e.graph < model->graphs.size()) {
    session.push_graph(e.graph);
  }

  int32_t node = node_index_for_hit(*model, e);
  if (node < 0) {
    app.view().search_open = false;
    return;
  }

  // Locate the display node; if it resolved to a collapsed group, expand it so
  // the leaf becomes visible, then re-resolve to the now-visible leaf.
  int32_t disp = panel_detail::display_index_for_node(session.collapse(),
                                                     static_cast<uint32_t>(node));
  const auto& display = session.collapse().display_nodes();
  if (disp >= 0 && static_cast<size_t>(disp) < display.size() &&
      display[disp].is_group && !display[disp].expanded) {
    session.toggle_group(display[disp].group_index);
    disp = panel_detail::display_index_for_node(session.collapse(),
                                               static_cast<uint32_t>(node));
  }

  if (disp >= 0) {
    app.view().selected_display = disp;
    // Fly-to: center the node in the current viewport. The canvas dock size is
    // not known from here, so use the main viewport work area as the view size
    // proxy — animate_camera_to only needs a reasonable extent to solve pan.
    panel_detail::BoxCenter center =
        panel_detail::box_center_for_display(session.layout(), disp);
    if (center.x != 0.0f || center.y != 0.0f) {
      ImGuiViewport* vp = ImGui::GetMainViewport();
      animate_camera_to(app.view(), ImVec2(center.x, center.y), vp->WorkSize,
                        app.view().cam.zoom);
    } else {
      // Layout not ready yet: leave selection set so the canvas can center on it
      // once a layout is published (request_fit is the closest existing intent).
      app.view().request_fit = false;
    }
  }
  app.view().search_open = false;
}

}  // namespace

// Draw the search overlay (spec §8.4). No-op unless the overlay is open.
void draw_search_bar(App& app) {
  ViewState& vs = app.view();
  // Latch so we grab focus + seed the buffer only on the frame the overlay
  // opens. Reset here (before any early-out) so it re-arms after each close.
  static bool was_open = false;
  if (!vs.search_open) {
    was_open = false;
    return;
  }

  const ir::Model* model = app.session().model();
  const SearchIndex& index = app.session().search();

  // Position near top-center of the main viewport.
  ImGuiViewport* vp = ImGui::GetMainViewport();
  ImVec2 pos(vp->WorkPos.x + vp->WorkSize.x * 0.5f, vp->WorkPos.y + 60.0f);
  ImGui::SetNextWindowPos(pos, ImGuiCond_Always, ImVec2(0.5f, 0.0f));
  ImGui::SetNextWindowSize(ImVec2(560.0f, 0.0f), ImGuiCond_Always);

  const ImGuiWindowFlags flags =
      ImGuiWindowFlags_NoTitleBar | ImGuiWindowFlags_NoResize |
      ImGuiWindowFlags_NoMove | ImGuiWindowFlags_NoCollapse |
      ImGuiWindowFlags_NoSavedSettings | ImGuiWindowFlags_NoDocking |
      ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs;

  if (!ImGui::Begin("##search_overlay", nullptr, flags)) {
    ImGui::End();
    return;
  }

  // Esc closes.
  if (ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    vs.search_open = false;
    ImGui::End();
    return;
  }

  // InputText over a char buffer synced to vs.search_query. imgui_stdlib is not
  // compiled into this build, so we can't use the std::string overload; sync a
  // fixed buffer each frame instead.
  static char buf[256];
  if (!was_open) {
    // Just opened: seed the buffer and grab keyboard focus (spec §8.4).
    std::snprintf(buf, sizeof(buf), "%s", vs.search_query.c_str());
    ImGui::SetKeyboardFocusHere();
  }
  was_open = true;

  ImGui::SetNextItemWidth(-FLT_MIN);
  bool changed = ImGui::InputText("##search_query", buf, sizeof(buf),
                                  ImGuiInputTextFlags_EnterReturnsTrue);
  // #52/#54: field-query syntax hint (kept subtle; the plain fuzzy path still works).
  ImGui::TextDisabled("op:Conv  name:layer*  dtype:fp16  shape:*224*  params:>1M");
  bool enter_pressed = changed;  // EnterReturnsTrue -> `changed` means submit
  // Detect edits separately (InputText w/ EnterReturnsTrue returns true only on
  // Enter, so compare buffer to the stored query to catch typing).
  if (std::strcmp(buf, vs.search_query.c_str()) != 0 && !enter_pressed) {
    vs.search_query = buf;
    vs.search_active_result = 0;
  } else if (enter_pressed) {
    vs.search_query = buf;
  }

  // #123: cached across frames, keyed on the query + the index + the model (see
  // QueryCache). Results still track the live buffer exactly — the query string
  // is part of the key, so a keystroke is a miss and recomputes — but holding a
  // key steady no longer costs a full index scan per frame. #52/#54: the
  // field-aware overload handles op:/name:/dtype:/shape:/params: predicates
  // against the live model, and falls back to the fuzzy path for a bare query.
  static QueryCache overlay_cache;
  static const std::vector<SearchHit> kNoHits;
  const std::vector<SearchHit>* hits_p = &kNoHits;
  if (model && !vs.search_query.empty()) {
    // types_ready gates dtype/shape/params predicates on shape inference having
    // published (else those fields are worker-mutated — a data race to read).
    // It is part of the cache key too: the same query legitimately returns more
    // once shapes land, and serving the pre-inference answer afterwards would be
    // the same stale-derived-state bug the cost report once had.
    const bool types_ready = app.session().stage() == LoadStage::Ready;
    hits_p = &cached_query(index, *model, vs.search_query, types_ready, 100,
                           overlay_cache);
  }
  const std::vector<SearchHit>& hits = *hits_p;

  // Keyboard navigation over the results.
  int nres = static_cast<int>(hits.size());
  if (nres > 0) {
    if (ImGui::IsKeyPressed(ImGuiKey_DownArrow))
      vs.search_active_result = (vs.search_active_result + 1) % nres;
    if (ImGui::IsKeyPressed(ImGuiKey_UpArrow))
      vs.search_active_result = (vs.search_active_result + nres - 1) % nres;
    if (vs.search_active_result >= nres) vs.search_active_result = nres - 1;
    if (vs.search_active_result < 0) vs.search_active_result = 0;
  }

  // Enter resolves the active result.
  if (enter_pressed && nres > 0 &&
      vs.search_active_result < static_cast<int>(hits.size())) {
    const SearchEntry& e = index.entries()[hits[vs.search_active_result].entry];
    resolve_hit(app, e);
    ImGui::End();
    return;
  }

  // Results list.
  if (nres > 0) {
    ImGui::Separator();
    ImVec2 list_size(0.0f, std::min(nres, 12) * ImGui::GetTextLineHeightWithSpacing() + 8.0f);
    if (ImGui::BeginChild("##results", list_size)) {
      for (int i = 0; i < nres; ++i) {
        const SearchHit& h = hits[i];
        if (h.entry >= index.entries().size()) continue;
        const SearchEntry& e = index.entries()[h.entry];
        ImGui::PushID(i);
        bool selected = (i == vs.search_active_result);
        std::string label = e.display.empty() ? "(anon)" : e.display;
        if (ImGui::Selectable(label.c_str(), selected,
                              ImGuiSelectableFlags_SpanAllColumns)) {
          vs.search_active_result = i;
          resolve_hit(app, e);
          ImGui::PopID();
          ImGui::EndChild();
          ImGui::End();
          return;
        }
        // Small right-aligned kind / graph tag.
        ImGui::SameLine();
        ImGui::TextDisabled("  [%s g%u]", kind_tag(e.kind), e.graph);
        if (selected) ImGui::SetScrollHereY(0.5f);
        ImGui::PopID();
      }
    }
    ImGui::EndChild();
  } else if (!vs.search_query.empty()) {
    ImGui::TextDisabled("No matches.");
  }

  ImGui::End();
}

// #53: persistent, dockable results panel — shows ALL hits for the shared query
// (not just the fly-to-first overlay). Shares vs.search_query with the Ctrl+F
// overlay, so typing in either updates both. Self-hides unless show_search_results.
void draw_search_results_panel(App& app) {
  ViewState& vs = app.view();
  if (!vs.show_search_results) return;

  if (!ImGui::Begin("Search results", &vs.show_search_results)) {
    ImGui::End();
    return;
  }

  const ir::Model* model = app.session().model();
  const SearchIndex& index = app.session().search();

  // Editable query, synced to the SAME vs.search_query the overlay uses.
  static char buf[256];
  std::snprintf(buf, sizeof(buf), "%s", vs.search_query.c_str());
  ImGui::SetNextItemWidth(-FLT_MIN);
  if (ImGui::InputTextWithHint("##results_query", "search (op:/name:/dtype:/...)",
                               buf, sizeof(buf)))
    vs.search_query = buf;
  ImGui::TextDisabled("op:Conv  name:layer*  dtype:fp16  params:>1M");
  ImGui::Separator();

  if (model == nullptr || vs.search_query.empty()) {
    ImGui::TextDisabled("Type a query above.");
    ImGui::End();
    return;
  }

  // Unbounded-ish list (cap generous); click any row to fly there. #123: its own
  // cache slot, because `limit` is part of the key — see QueryCache for why the
  // 500-hit answer cannot be truncated to serve the overlay's 100.
  const bool types_ready = app.session().stage() == LoadStage::Ready;
  static QueryCache panel_cache;
  const std::vector<SearchHit>& hits =
      cached_query(index, *model, vs.search_query, types_ready, 500,
                   panel_cache);
  ImGui::TextDisabled("%zu match%s", hits.size(), hits.size() == 1 ? "" : "es");

  if (ImGui::BeginChild("##results_list")) {
    for (size_t i = 0; i < hits.size(); ++i) {
      if (hits[i].entry >= index.entries().size()) continue;
      const SearchEntry& e = index.entries()[hits[i].entry];
      ImGui::PushID(static_cast<int>(i));
      // #123: no per-row std::string. e.display is already a std::string owned
      // by the index, so copying it once per visible row per frame bought
      // nothing but an allocation.
      const char* label = e.display.empty() ? "(anon)" : e.display.c_str();
      if (ImGui::Selectable(label, false,
                            ImGuiSelectableFlags_SpanAllColumns)) {
        // resolve_hit clears search_open (overlay) but leaves this panel open.
        resolve_hit(app, e);
      }
      ImGui::SameLine();
      ImGui::TextDisabled("  [%s g%u]", kind_tag(e.kind), e.graph);
      ImGui::PopID();
    }
  }
  ImGui::EndChild();

  ImGui::End();
}

}  // namespace netvis
