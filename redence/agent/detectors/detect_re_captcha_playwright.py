from playwright.sync_api import Error, Page

from selenium_captcha_processing.config import Config
from selenium_captcha_processing.utils.container import Utils


class DetectReCaptchaPlaywright:
    def __init__(self, page: Page, utils: Utils, config: Config, *args, **kwargs):
        self.config = config
        self.utils = utils
        self.page = page

    def detected(self) -> float:
        score = 0.0
        timeout_ms = int(self.config.default_element_waiting * 1000)

        recaptcha_iframe = self.page.locator('iframe[title="reCAPTCHA"]').first
        if recaptcha_iframe.count() > 0 and recaptcha_iframe.is_visible(timeout=timeout_ms):
            score += 0.25

        site_key = self.page.locator('[data-sitekey]').first
        if site_key.count() > 0 and site_key.is_visible(timeout=timeout_ms):
            score += 0.5

        try:
            js_obj = self.page.evaluate("() => typeof window.grecaptcha !== 'undefined'")
            if js_obj:
                score += 0.25
        except Error:
            # Keep detector non-fatal if script execution is blocked.
            pass

        return score
