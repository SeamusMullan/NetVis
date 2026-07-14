// main.cpp — process entry point.
//
// Creates the App, forwards an optional CLI path (argv[1]) as the initial file,
// and runs the main loop. All real work lives in App; keeping main tiny means
// the same App can be driven from a test harness without a second entry point.
#include <string>

// LayoutEngine.h defines SizeFn, referenced by the frozen ModelSession.h (via
// App.h) but not included there; pre-include it so App.h compiles.
#include "engine/LayoutEngine.h"
#include "view/App.h"

int main(int argc, char** argv) {
  netvis::App app;
  std::string path = argc > 1 ? argv[1] : std::string();
  if (!app.init(path)) return 1;
  return app.run();
}
