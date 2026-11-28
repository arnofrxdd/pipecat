#
# Copyright (c) 2024-2026, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#

import pytest

pytest.importorskip("sarvamai")

from pipecat.services.sarvam.stt import SarvamSTTService, language_to_sarvam_language
from pipecat.transcriptions.language import Language


@pytest.mark.parametrize(
    "language, expected",
    [
        (Language.HI_IN, "hi-IN"),
        (Language.UR_IN, "ur-IN"),
        (Language.KOK_IN, "kok-IN"),
        (Language.MAI_IN, "mai-IN"),
        (Language.SD_IN, "sd-IN"),
    ],
)
def test_language_to_sarvam_language_maps_enum_values(language, expected):
    assert language_to_sarvam_language(language) == expected


@pytest.mark.parametrize("language_code", ["ne-IN", "sat-IN"])
def test_get_language_string_passes_through_string_values(language_code):
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = language_code

    assert service._get_language_string() == language_code


def test_get_language_string_resolves_enum_via_mapping():
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = Language.HI_IN

    assert service._get_language_string() == "hi-IN"


def test_get_language_string_returns_model_default_when_unset():
    service = SarvamSTTService(api_key="test-key")
    service._settings.language = None

    assert service._get_language_string() == service._config.default_language
    assert service._config.default_language == "unknown"


def test_prepare_audio_chunk_prepends_wav_header_only_on_first_chunk():
    service = SarvamSTTService(api_key="test-key", sample_rate=16000)
    assert service._needs_wav_header is True

    pcm_chunk_1 = b"\x00\x00" * 160  # 10ms of silence
    pcm_chunk_2 = b"\x12\x34" * 160

    prepared_1 = service._prepare_audio_chunk(pcm_chunk_1)
    # Header should be 44 bytes + 320 bytes PCM = 364 bytes
    assert len(prepared_1) == 44 + len(pcm_chunk_1)
    assert prepared_1.startswith(b"RIFF")
    assert b"WAVEfmt " in prepared_1
    assert b"data" in prepared_1
    assert service._needs_wav_header is False

    # Second chunk must NOT have the header prepended again
    prepared_2 = service._prepare_audio_chunk(pcm_chunk_2)
    assert len(prepared_2) == len(pcm_chunk_2)
    assert prepared_2 == pcm_chunk_2


def test_prepare_audio_chunk_does_not_duplicate_existing_riff_header():
    service = SarvamSTTService(api_key="test-key")
    existing_wav = b"RIFF" + b"\x00" * 40 + b"audio_data"
    result = service._prepare_audio_chunk(existing_wav)
    assert result == existing_wav

