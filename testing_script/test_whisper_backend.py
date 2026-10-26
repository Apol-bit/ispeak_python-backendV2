"""Backend selection and local loading contracts; no weights or speech inference."""
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock,patch

from model import configured_whisper_backend,ISpeakWhisper,ModelUnavailableError


class WhisperBackendTests(unittest.TestCase):
    def test_default_remains_adapted_and_unknown_configuration_fails_closed(self):
        self.assertEqual(configured_whisper_backend({}),'adapted')
        self.assertEqual(configured_whisper_backend({'ISPEAK_WHISPER_BACKEND':'base'}),'base')
        for value in ('','auto','remote','PRIVATE_TOKEN_OR_PATH'):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ModelUnavailableError,'^Invalid Whisper transcription backend$'):
                    configured_whisper_backend({'ISPEAK_WHISPER_BACKEND':value})

    def load(self,backend):
        base=SimpleNamespace(eval=Mock(),device='cpu')
        merged=SimpleNamespace(eval=Mock(),device='cpu')
        processor=SimpleNamespace(tokenizer=object(),feature_extractor=object())
        adapter=SimpleNamespace(merge_and_unload=Mock(return_value=merged))
        pipe=SimpleNamespace()
        with ExitStack() as stack:
            stack.enter_context(patch.dict('os.environ',{'ISPEAK_WHISPER_BACKEND':backend,'ISPEAK_ENV':'production'}))
            stack.enter_context(patch('model.resolve_model_paths',return_value=(Path('/adapter'),Path('/base'))))
            stack.enter_context(patch('model.declared_base_model_id',return_value='openai/whisper-small'))
            stack.enter_context(patch('torch.cuda.is_available',return_value=False))
            load_processor=stack.enter_context(patch('transformers.WhisperProcessor.from_pretrained',return_value=processor))
            load_base=stack.enter_context(patch('transformers.WhisperForConditionalGeneration.from_pretrained',return_value=base))
            load_adapter=stack.enter_context(patch('peft.PeftModel.from_pretrained',return_value=adapter))
            make_pipe=stack.enter_context(patch('transformers.pipeline',return_value=pipe))
            hooks=[stack.enter_context(patch(target,return_value=True)) for target in [
                'audio_analysis_processing_files.whisper_timestamps.install_selected_head_timestamps',
                'audio_analysis_processing_files.whisper_retention.install_longform_retention',
                'audio_analysis_processing_files.whisper_retention.install_active_chunk_release',
                'audio_analysis_processing_files.whisper_fallback.install_fallback_diagnostics']]
            model=ISpeakWhisper()
        load_processor.assert_called_once_with(str(Path('/adapter')),local_files_only=True)
        load_base.assert_called_once_with(str(Path('/base')),local_files_only=True)
        self.assertIs(model.processor,processor)
        self.assertIs(model.pipe,pipe)
        for hook in hooks:
            self.assertIs(hook.call_args.args[0],model.model)
        make_pipe.assert_called_once_with('automatic-speech-recognition',model=model.model,
            tokenizer=processor.tokenizer,feature_extractor=processor.feature_extractor,device=-1)
        return model,base,merged,load_adapter,adapter

    def test_base_uses_unmodified_local_weights_without_loading_or_merging_lora(self):
        model,base,merged,load_adapter,adapter=self.load('base')
        self.assertIs(model.model,base)
        base.eval.assert_called_once_with()
        merged.eval.assert_not_called()
        load_adapter.assert_not_called()
        adapter.merge_and_unload.assert_not_called()
        self.assertEqual(model.model_name,'whisper-small-base')

    def test_adapted_keeps_original_safe_merge_and_readonly_loading(self):
        model,base,merged,load_adapter,adapter=self.load('adapted')
        self.assertIs(model.model,merged)
        merged.eval.assert_called_once_with()
        load_adapter.assert_called_once_with(base,str(Path('/adapter')),is_trainable=False,local_files_only=True)
        adapter.merge_and_unload.assert_called_once_with(safe_merge=True)

    def test_invalid_backend_does_not_attempt_loading_or_expose_configuration_value(self):
        with patch.dict('os.environ',{'ISPEAK_WHISPER_BACKEND':'PRIVATE_VALUE'}),patch('model.resolve_model_paths') as paths:
            with self.assertRaises(ModelUnavailableError) as caught:ISpeakWhisper()
        paths.assert_not_called()
        self.assertNotIn('PRIVATE_VALUE',str(caught.exception))


if __name__=='__main__':unittest.main()
