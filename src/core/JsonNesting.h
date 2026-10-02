// SPDX-License-Identifier: Apache-2.0
// core/JsonNesting.h — brace/bracket nesting pre-scan for untrusted JSON text.
//
// nlohmann::json::parse is recursive and has no depth limit of its own, so a file
// that is nothing but `[[[[...` can overflow the stack. Callers that parse JSON
// from an untrusted file run this scan FIRST and refuse the file when it fails.
// Promoted from the plugin-manifest loader (declarative/Manifest.cpp) so the
// .netvis-view reader (#170) shares exactly the same check.
#pragma once

#include <cstddef>
#include <string_view>

namespace netvis {

// Returns false when the text nests more than `max_depth` objects/arrays deep.
// Brackets inside string literals (including after an escaped quote) do not
// count, and an unbalanced closer never drives the depth negative. This is a
// cheap pre-filter, not a validator: malformed JSON that passes it is still
// rejected by the real parser afterwards.
//
// `allow_comments` MUST match how the text is parsed. A caller that parses with
// nlohmann's `ignore_comments = true` (the plugin manifests) has to pass true: the
// parser then skips `// ...` (to the end of the line) and `/* ... */`, and so must
// this scan, or a quote inside a comment flips it into string mode and the real
// brackets after it are never counted (a deep document then passes this check and
// can overflow the stack in any recursive operation on the parsed value). A caller
// whose parser rejects comments (the .netvis-view reader) leaves it false: the text
// is then refused at the first `/` anyway, so how it is scanned does not matter.
inline bool json_nesting_ok(std::string_view text, int max_depth, bool allow_comments = false) {
  int depth = 0;
  bool in_str = false, esc = false;
  const size_t n = text.size();
  for (size_t i = 0; i < n; ++i) {
    const char c = text[i];
    if (in_str) {
      if (esc) esc = false;
      else if (c == '\\') esc = true;
      else if (c == '"') in_str = false;
      continue;
    }
    if (allow_comments && c == '/' && i + 1 < n) {
      if (text[i + 1] == '/') {
        // Same terminators as nlohmann's lexer: newline, carriage return, NUL.
        i += 2;
        while (i < n && text[i] != '\n' && text[i] != '\r' && text[i] != '\0') ++i;
        continue;  // the terminator itself is whitespace (or ends the scan)
      }
      if (text[i + 1] == '*') {
        // To the closing "*/". An unterminated comment (or a NUL in one) is a parse
        // error for nlohmann, so there is nothing left to count.
        const size_t end = text.find("*/", i + 2);
        if (end == std::string_view::npos) return true;
        i = end + 1;
        continue;
      }
    }
    if (c == '"') in_str = true;
    else if (c == '{' || c == '[') { if (++depth > max_depth) return false; }
    else if (c == '}' || c == ']') { if (depth > 0) --depth; }
  }
  return true;
}

}  // namespace netvis
