// mcp_main.cpp — entry point for the MCP server.
//
// Links netvis_core ONLY: no GLFW, no OpenGL, no ImGui, no window. MCP clients
// spawn their servers over stdio and own the process for the session, so this
// binary is what an agent's server configuration points at — especially on
// machines with no display stack. The GUI binary (`netvis mcp`) and the query
// binary (`netvis_query mcp`) start the identical loop, so any of the three
// works interchangeably in a client configuration.
#include "engine/McpServer.h"

int main() {
  return netvis::run_mcp_stdio();
}
