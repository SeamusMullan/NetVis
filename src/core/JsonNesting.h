// SPDX-License-Identifier: Apache-2.0
// core/JsonNesting.h — brace/bracket nesting pre-scan for untrusted JSON text.
//
// nlohmann::json::parse is recursive and has no depth limit of its own, so a file
// that is nothing but `[[[[...` can overflow the stack. Callers that parse JSON
// from an untrusted file run this scan FIRST and refuse the file when it fails.
// Promoted from the plugin-manifest loader (declarative/Manifest.cpp) so the
// .netvis-view reader (#170) shares exactly the same check.
#pragma once

#include <string_view>

namespace netvis {

// Returns false when the text nests more than `max_depth` objects/arrays deep.
// Brackets inside string literals (including after an escaped quote) do not
// count, and an unbalanced closer never drives the depth negative. This is a
// cheap pre-filter, not a validator: malformed JSON that passes it is still
// rejected by the real parser afterwards.
inline bool json_nesting_ok(std::string_view text, int max_depth) {
  int depth = 0;
  bool in_str = false, esc = false;
  for (char c : text) {
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (c == '"') in_str = true;
    else if (c == '{' || c == '[') { if (++depth > max_depth) return false; }
    else if (c == '}' || c == ']') { if (depth > 0) --depth; }
  }
  return true;
}

}  // namespace netvis
