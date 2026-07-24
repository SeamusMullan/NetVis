// tests/test_plugins_enable.cpp — per-plugin enable/disable + trust gate (#11).
//
// Core-only (no view): exercises PluginEnableSet defaults/overrides/persistence and
// the gated discovery -> table-membership behavior (a disabled plugin is structurally
// absent from the Registry, §0.4). WASM cases are covered by test_wasm_parser; here
// we test the DECLARATIVE gate + the enable-set semantics that need no sandbox.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "engine/plugin/PluginPrefs.h"
#include "engine/plugin/Registry.h"
#include "engine/plugin/declarative/Manifest.h"

using namespace netvis;
using plugin::PluginEnableSet;
using plugin::PluginKind;

TEST_CASE("PluginEnableSet: defaults, overrides, never-prune round trip") {
  PluginEnableSet s;
  // Absent key -> kind default (declarative on, wasm off).
  CHECK(s.effective("a", PluginKind::Declarative) == true);
  CHECK(s.effective("b", PluginKind::Wasm) == false);
  CHECK_FALSE(s.has_explicit("a"));

  s.set("a", false);   // disable a declarative
  s.set("b", true);    // enable a wasm
  CHECK(s.effective("a", PluginKind::Declarative) == false);
  CHECK(s.effective("b", PluginKind::Wasm) == true);
  CHECK(s.has_explicit("a"));

  // Round-trip through JSON; entries never pruned.
  nlohmann::json j = s.to_json();
  PluginEnableSet s2;
  s2.load_json(j);
  CHECK(s2.effective("a", PluginKind::Declarative) == false);
  CHECK(s2.effective("b", PluginKind::Wasm) == true);

  // Non-object / non-bool inputs are ignored, never throw.
  PluginEnableSet s3;
  s3.load_json(nlohmann::json::array({1, 2}));       // not an object
  s3.load_json(nlohmann::json{{"x", "notbool"}});     // non-bool value
  CHECK_FALSE(s3.has_explicit("x"));
  CHECK(s3.effective("x", PluginKind::Declarative) == true);   // still default
}

namespace {
// Write a minimal declarative plugin dir under `root/<id>/plugin.json`.
void write_decl_plugin(const std::string& root, const std::string& id,
                       const std::string& op_name) {
  std::filesystem::create_directories(root + "/" + id);
  nlohmann::json j;
  j["api_version"] = plugin::kOpHandlerAbiVersion;
  j["name"] = id;
  nlohmann::json op;
  op["name"] = op_name;      // a CUSTOM op name (not a built-in) so it registers w/o override
  op["category"] = "Other";
  j["ops"] = nlohmann::json::array({op});
  std::ofstream f(root + "/" + id + "/plugin.json");
  f << j.dump(2);
}
}  // namespace

TEST_CASE("declarative gate: disabled plugin is structurally absent from Registry") {
  namespace fs = std::filesystem;
  fs::path root = fs::temp_directory_path() / "nv_plugtest";
  fs::remove_all(root);
  write_decl_plugin(root.string(), "aa", "MyGateAOp");
  write_decl_plugin(root.string(), "bb", "MyGateBOp");

  plugin::Registry& reg = plugin::Registry::instance();
  reg.reset_to_builtins();

  // Gate: enable "aa", disable "bb".
  auto gate = [](std::string_view id, PluginKind) { return id == "aa"; };
  auto manifests = plugin::discover_and_load_plugins(gate, root.string());

  // Both discovered (for the panel); one registered, one not.
  REQUIRE(manifests.size() == 2);
  // aa registered -> its op resolves as Declarative; bb absent -> Builtin catch-all.
  CHECK(reg.resolve_op("mygateaop").origin == plugin::Origin::Declarative);
  CHECK(reg.resolve_op("mygatebop").origin == plugin::Origin::Builtin);

  // The panel still SEES both (diagnostics populated), enabled flag reflects the gate.
  bool saw_aa_enabled = false, saw_bb_disabled = false;
  for (const auto& lm : manifests) {
    if (lm.id == "aa") saw_aa_enabled = lm.enabled && lm.registered;
    if (lm.id == "bb") saw_bb_disabled = !lm.enabled && !lm.registered;
  }
  CHECK(saw_aa_enabled);
  CHECK(saw_bb_disabled);

  reg.reset_to_builtins();
  fs::remove_all(root);
}

TEST_CASE("reset_to_builtins drops a registered declarative plugin") {
  namespace fs = std::filesystem;
  fs::path root = fs::temp_directory_path() / "nv_plugtest2";
  fs::remove_all(root);
  write_decl_plugin(root.string(), "cc", "MyResetOp");

  plugin::Registry& reg = plugin::Registry::instance();
  reg.reset_to_builtins();
  plugin::discover_and_load_plugins([](std::string_view, PluginKind) { return true; },
                                    root.string());
  CHECK(reg.resolve_op("myresetop").origin == plugin::Origin::Declarative);

  reg.reset_to_builtins();
  CHECK(reg.resolve_op("myresetop").origin == plugin::Origin::Builtin);
  fs::remove_all(root);
}
