// engine/plugin/PassPlugin.h — FROZEN analysis-pass ABI (v0.6.0, issue #7).
//
// A PassPlugin computes named metrics over an already-parsed model + its
// CostReport. PURE (no payload): sees ir::Model structure + the engine CostReport
// (itself produced under inv 3), never a weight buffer. CostReport lives in
// engine/CostModel.h (ir + engine-leaf, view-free), so including it adds no UI dep.
#pragma once

#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "engine/CostModel.h"   // CostReport
#include "ir/IR.h"

namespace netvis::plugin {

inline constexpr uint32_t kPassPluginAbiVersion = 1;

struct Metric {                 // known=false renders as "—" (honest-unknown)
  std::string name;
  double      value = 0.0;
  bool        known = false;
  std::string unit;             // free label ("FLOP","MB","%")
};
struct PassResult { std::vector<Metric> metrics; };  // may be empty

class PassPlugin {
 public:
  virtual ~PassPlugin() = default;
  virtual std::string_view display_name() const = 0;
  // graph_index is the graph the report was built for. Never throws; no payload.
  virtual PassResult run(const ir::Model& model, uint32_t graph_index,
                         const CostReport& report) const = 0;
  virtual uint32_t api_version() const = 0;
};

}  // namespace netvis::plugin
