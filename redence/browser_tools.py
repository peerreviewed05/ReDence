"""Controller adapter for the existing Playwright investigation tools."""
import importlib.util
import json
from pathlib import Path


class BrowserTools:
    def __init__(self, output_dir):
        # Load directly: agent/__init__.py belongs to the vendored CAPTCHA package.
        path = Path(__file__).parent / "agent/browser_agent.py"
        spec = importlib.util.spec_from_file_location("redence_browser_impl", path)
        self.impl = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.impl)
        self.output_dir = Path(output_dir)
        self.step = [0]

    def __enter__(self):
        m = self.impl
        try:
            pw = m.sync_playwright().start()
            m.SESSION["playwright"] = pw
            browser = m._launch_browser(pw)
            m.SESSION["browser"] = browser
            context = browser.new_context(viewport={"width": 1920, "height": 1080},
                user_agent=m.BROWSER_HEADERS["User-Agent"], ignore_https_errors=True)
            m.SESSION["context"] = context
            context.set_default_navigation_timeout(m.NAV_TIMEOUT)
            if m._env_bool("PLAYWRIGHT_STEALTH", True):
                context.add_init_script(m.STEALTH_INIT_SCRIPT)
            m.SESSION["page"] = context.new_page()
            return self
        except Exception:
            self.close()
            raise

    def close(self):
        self.impl._cleanup_browser_session()

    def __exit__(self, *args):
        self.close()

    def execute(self, name, arguments):
        m = self.impl
        kwargs = {"screens_dir": self.output_dir, "step_ref": self.step}
        if name == "web_search":
            return {"results": m.search_web(arguments["query"])}, None
        if name == "fetch_page":
            result = m.fetch_page(arguments["url"], **kwargs)
        elif name == "interact_page":
            result = m.interact_page(arguments.get("url"), arguments, **kwargs)
        elif name == "model_take_screenshot":
            capture = m.take_screenshot(arguments.get("label", "snapshot"), **kwargs)
            image = capture.get("vision_b64_jpeg")
            return m._tool_result_from_capture(capture, vision_delivered=bool(image)), image
        else:
            raise ValueError(f"Unknown browser tool: {name}")
        if isinstance(result, str):
            try:
                result = json.loads(result)
            except json.JSONDecodeError:
                result = {"text": result}
        if result.get("error"):
            return result, None
        page = m.SESSION["page"]
        return {"result": result, "url": page.url, "marks": m.extract_marks(page)}, None
