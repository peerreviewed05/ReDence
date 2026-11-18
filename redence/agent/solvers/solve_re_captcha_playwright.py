import os
import tempfile
import uuid
from time import sleep

from playwright.sync_api import Locator, Page, TimeoutError as PlaywrightTimeoutError
from pydub import AudioSegment

from selenium_captcha_processing.config import Config
from selenium_captcha_processing.helpers import download_audio
from selenium_captcha_processing.utils.container import Utils


class SolveReCaptchaPlaywright:
    def __init__(self, page: Page, utils: Utils, config: Config, *args, **kwargs):
        self.config = config
        self.utils = utils
        self.page = page

    def solve(self) -> bool:
        timeout_ms = int(self.config.default_element_waiting * 1000)

        checkbox = self.page.frame_locator('iframe[title="reCAPTCHA"]').locator("#recaptcha-anchor")
        if not self._wait_visible(checkbox, timeout_ms):
            return False

        checkbox.click()
        if checkbox.get_attribute("aria-checked") == "true":
            return True

        return self._solve_challenge()

    def _solve_challenge(self) -> bool:
        challenge_frame = self.page.frame_locator('iframe[src*="recaptcha"][src*="bframe"]')
        audio_btn = challenge_frame.locator("#recaptcha-audio-button")
        if not self._wait_visible(audio_btn, 1500):
            return False
        audio_btn.click()

        download_link = challenge_frame.locator(".rc-audiochallenge-tdownload-link")
        if not self._wait_visible(download_link, 7000):
            return False

        link = download_link.get_attribute("href")
        if not link:
            return False

        tmp_dir = tempfile.gettempdir()
        audio_file_id = uuid.uuid4().hex
        mp3_file = os.path.join(tmp_dir, f"{audio_file_id}_tmp.mp3")
        wav_file = os.path.join(tmp_dir, f"{audio_file_id}_tmp.wav")

        try:
            download_audio(link, mp3_file)
            AudioSegment.from_mp3(mp3_file).export(wav_file, format="wav")
            recognized_text = self.utils.speech_recogniser.recognise_from_file(wav_file)
        finally:
            for path in (mp3_file, wav_file):
                if os.path.exists(path):
                    os.remove(path)

        if not recognized_text:
            return False

        response_textbox = challenge_frame.locator("#audio-response")
        if not self._wait_visible(response_textbox, int(self.config.default_element_waiting * 1000)):
            return False
        self._human_type(response_textbox, recognized_text)

        verify_button = challenge_frame.locator("#recaptcha-verify-button")
        if not self._wait_visible(verify_button, 5000):
            return False
        verify_button.click()

        # Give recaptcha a short moment to update anchor state.
        sleep(0.4)
        checkbox = self.page.frame_locator('iframe[title="reCAPTCHA"]').locator("#recaptcha-anchor")
        return checkbox.get_attribute("aria-checked") == "true"

    @staticmethod
    def _wait_visible(locator: Locator, timeout_ms: int) -> bool:
        try:
            locator.wait_for(state="visible", timeout=timeout_ms)
            return True
        except PlaywrightTimeoutError:
            return False

    @staticmethod
    def _human_type(locator: Locator, text: str) -> None:
        locator.click()
        locator.press_sequentially(text, delay=30)
