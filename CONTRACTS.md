# NetVis — Frozen Interface Contracts

These headers are FROZEN. Implementations must match exactly; do not change a
signature without updating every caller. All types live in `namespace netvis`
(IR in `netvis::ir`). Include root is `src/`.

## Layers
```
view/  -> engine/ (ModelSession only)  -> parsers/ -> ir::Model
                  engine also: LayoutEngine, CollapseTree, SearchIndex,
                  ShapeInference, TensorStats, LayoutCache
core/  is shared by all (Result, MappedFile, ByteReader, StringArena, Hash,
       JobSystem, SmallVec).
```

## core/ (DONE, frozen)
- `Result<T>` — `ok()`, `*`, `.error()`, `.take()`; `err(msg,offset)`.
- `MappedFile::open(path) -> Result<MappedFile>`; `.data()/.size()/.path()`.
- `ByteReader(data,size)` — `u8/u16le/u32le/u64le/i32le/i64le/f32le/f64le`,
  `bytes(n)`, `raw(n)`, `seek/skip/remaining/pos`. All bounds-checked → `Result`.
  `ByteReader::mark_payload_read()` + `payload_read_counter()` for tests.
- `StringArena` — `intern(sv) -> StringId`, `get(id) -> string_view`. id 0 = "".
- `Hasher` (FNV-1a) — `.u64/.u32/.i64/.str().value()`; `fnv1a(sv)`.
- `JobSystem` — `submit(fn)`, `post_to_main(fn)`, `drain_completions()` (main),
  `generation()/bump_generation()`. `ProgressSink::set(frac,stage)`.

## ir::Model (DONE, frozen) — ir/IR.h
`DType` enum; `dtype_name`/`dtype_size`. `Range{begin,count}`.
`TensorRef{name,dtype,shape,file_offset,byte_len,external_path}` (offset=UINT64_MAX
if absent); `.elem_count()`. `AttrValue{kind,i,f,s,ints,floats,strings,tensor,graph}`.
`Attribute{name,value}`. `Node{op_type,name,inputs,outputs,attributes,subgraph}`.
`ValueInfo{name,dtype,shape,producer}`. `Graph{name,nodes,values,edge_refs,
attributes,initializers,graph_inputs,graph_outputs}`. `Model{graphs,metadata,
format_name,producer,version_info,has_graph,flat_tensors,strings}` + `intern/str`.
edge_refs: a Node's inputs/outputs Range indexes into `Graph::edge_refs`, whose
entries are indices into `Graph::values`.

## parsers/ (TO IMPLEMENT) — parsers/Parser.h
- `Format` enum; `format_name(f)`; `detect_format(file, ext_hint) -> Format`.
- `parse_model(file, ext_hint, progress) -> Result<ir::Model>` (detect+dispatch).
- Per-format: `onnx::parse`, `tflite::parse`, `safetensors::parse`, `gguf::parse`,
  `pytorch::parse_zip`, `pytorch::parse_legacy` — all `(MappedFile&,ProgressSink&)
  -> Result<ir::Model>`.
- MUST: read only through ByteReader; never read tensor payloads (record
  offset+len); 64-bit offsets; malformed input → Result error w/ offset, no crash.

## engine/ (TO IMPLEMENT)
- **OpCategory.h**: `categorize_op(sv) -> OpCategory`; `category_name(c)`.
- **Layout.h** (types, frozen): `Vec2`, `NodeBox{display_id,pos,size,layer}`,
  `EdgeCurve{from,to,p0..p3,reversed}`, `LayoutResult{boxes,edges,bounds_min/max,
  structure_hash,collapse_hash,from_cache}`.
- **CollapseTree.h**: `CollapseTree::build(model,graph_idx)`, `groups()`,
  `display_nodes()`, `toggle_group(i)->bool`, `collapse_hash()`, `structure_hash()`.
  `node_fingerprint(model,g,node)->u64`. `DisplayNode{is_group,group_index,
  ir_node,expanded}`. `CollapseGroup{label,instances,member_nodes,
  representative_nodes,fingerprint}`.
- **LayoutEngine.h**: `SizeFn = function<Vec2(const DisplayNode&)>`.
  `compute_layout(model,graph_idx,collapse,size_fn,params,progress*) ->
  LayoutResult`. Deterministic.
- **SearchIndex.h**: `SearchIndex::build(model)`, `query(q,limit)->vector<SearchHit>`,
  `entries()`. `fuzzy_score(q_lower,text_lower)->int` (-1 = no match).
  `SearchEntry{lower,display,kind,graph,ref}`, `SearchHit{entry,score}`.
- **ShapeInference.h**: `infer_shapes(model, graph_idx, progress*) -> uint32_t`
  (mutates ValueInfo shapes in place; ONNX only).
- **TensorStats.h**: `compute_tensor_stats(tref,base,model_dir) -> Result<TensorStats>`;
  `export_npy(...)`, `export_raw(...) -> Result<bool>`. `TensorStats{min,max,mean,
  std,zero_count,nan_inf_count,count,histogram[64],hist_min/max,quantized_unsupported}`.
  This is the ONLY code that reads payloads → calls `mark_payload_read()`.
- **LayoutCache.h**: `layout_cache_dir()`, `load_cached_layout(sh,ch)`,
  `store_cached_layout(layout)`.
- **ModelSession.h**: `ModelSession(JobSystem&)`; `open_async(path)`, `update()`
  (main, per frame), `stage()`, `model()`, `has_graph()`, `layout()`, `collapse()`,
  `search()`, `file()`, `model_dir()`, `set_size_fn(fn)`, `toggle_group(i)`,
  `push_graph/pop_graph/graph_stack`, `timings()`, `progress()`, `generation()`.

## view/ (TO IMPLEMENT) — Dear ImGui docking, GLFW+GL3.3
- `App` owns GLFW window, JobSystem, ModelSession, dockspace, theme, panels.
- Widgets: GraphCanvas (ImDrawList only, culling, LOD), PropertiesPanel,
  WeightInspector, SearchBar, Minimap, TensorTable (has_graph==false), StatusBar.
- main.cpp: create App, handle CLI arg + GLFW drop callback, run loop
  (`session.update()` once/frame).

## v0.2.0 additions (additive — no frozen signature changed)
- **engine/GraphAdjacency.h** — CSR forward/reverse adjacency over one graph;
  `build(model,graph)`, `reachable_succ/pred(start,max_hops,cap)`. Built
  synchronously on the main thread (cheap O(V+E)).
- **engine/ModelDiff.h** — `diff_models(A,gA,B,gB) -> ModelDiffResult`
  (`a_status/b_status/a_to_b/b_to_a` + counts). Matches by string CONTENT across
  independent arenas; reads no shape/dtype. Deterministic.
- **engine/DiffLoader.h** — owns the comparison model + async load + diff (engine
  may include parsers; the view must not). Uses its OWN JobSystem.
- **engine/ShapeInferenceExt.h** — `infer_shapes_ext(model,graph,base,size,prog)`;
  the frozen 3-arg `infer_shapes` delegates here with `base=nullptr`.
- **view/GraphNav.h**, **view/DiffPanel.h** — module-private view state + panels.
- **`LayoutResult.boxes` invariant RELAXED:** after source duplication, `boxes`
  may exceed `display_nodes().size()` and several boxes may share a `display_id`
  (a clone carries its source's id). Consumers MUST key off `box.display_id`
  (bounds-checked `< display_nodes().size()`), never the box index. Any
  position-affecting layout change bumps `kVersion` in `LayoutCache.cpp`.
- **`view/` is NOT frozen.** This file (not the `App.h` banner comment) is the
  authority; `view/` is *to-implement*. `ViewState` is append-only. Frozen `App.h`
  panel `draw_*` signatures are unchanged; new panels are new free functions.

## Rules for implementers
- C++20, warnings-as-errors clean. RAII, no UB, no data races.
- Comment every perf-relevant decision at its site.
- Add source files freely (GLOB picks them up). Do NOT edit frozen headers.
- Each parser/engine module gets its own .cpp under its directory.
