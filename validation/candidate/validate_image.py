"""Image integrity and guarded model-free regression checks only."""
import hashlib
import json


def validate(v, tests, manifest):
    v.phase = 'dependency_inventory'
    inventory_code = "import importlib.metadata as m,json;print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"
    # Model-free metadata query; explicitly override the inherited integrity/server entrypoint.
    from materialize import BASE
    baseline = json.loads(v.docker(*v.run_args('base-inventory-' + v.key),
                                  '--entrypoint=python', BASE, '-B', '-c', inventory_code))
    candidate = json.loads(v.docker(*v.run_args('candidate-inventory-' + v.key),
                                   '--entrypoint=python', v.image, '-B', '-c', inventory_code))
    targets = {'certifi': '2026.10.19', 'filelock': '4.2.1', 'setuptools': '84.1.0'}
    expected = dict(baseline, **targets)
    if candidate != expected or candidate.get('accelerate') != '1.14.0':
        raise RuntimeError('Unexpected dependency drift or security version mismatch')
    hashes = {item['path']: item['sha256'] for item in manifest['files']
              if item['path'].endswith('.py')}
    code = r'''
import hashlib,json,os,pathlib,subprocess,sys,unittest
from unittest.mock import patch
expected=json.loads(sys.argv[1]); model_manifest_hash=sys.argv[2]
root=pathlib.Path('/app')
for name,digest in expected.items():
 assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest
assert not any((root/name).exists() for name in ('.env','.git','recordings'))
manifest=root/'model_manifests/ispeak-v6.json'
assert hashlib.sha256(manifest.read_bytes()).hexdigest()==model_manifest_hash
assert len(json.loads(manifest.read_text())['files'])==12
subprocess.run([sys.executable,'-B','/app/model_integrity.py'],check=True,capture_output=True)
subprocess.run([sys.executable,'-B','-m','pip','check'],check=True,capture_output=True)
import model
from audio_analysis_processing_files import filler_words
from testing_script.test_whisper_timestamp_memory import owner
from audio_analysis_processing_files.whisper_timestamps import install_selected_head_timestamps,_extract_selected
synthetic=owner()
assert install_selected_head_timestamps(synthetic), 'Timestamp optimization inactive'
assert synthetic._extract_token_timestamps.__func__ is _extract_selected
from testing_script.test_whisper_longform_retention import owner as retention_owner
from audio_analysis_processing_files.whisper_retention import install_longform_retention,_generate_with_fallback
from types import MethodType
from transformers.models.whisper.generation_whisper import WhisperGenerationMixin
from transformers.pipelines.automatic_speech_recognition import AutomaticSpeechRecognitionPipeline
retention=retention_owner()
assert install_longform_retention(retention,AutomaticSpeechRecognitionPipeline), 'Long-form retention optimization inactive'
assert retention.generate_with_fallback.__func__ is _generate_with_fallback
retention._need_fallback=MethodType(WhisperGenerationMixin._need_fallback,retention)
from audio_analysis_processing_files.whisper_fallback import install_fallback_diagnostics,_generate_with_diagnostics
assert install_fallback_diagnostics(retention), 'Bounded fallback diagnostics inactive'
assert retention.generate_with_fallback.__func__ is _generate_with_diagnostics
import torch
from transformers.models.whisper.modeling_whisper import WhisperForConditionalGeneration
from audio_analysis_processing_files.whisper_retention import install_active_chunk_release,_release_self_attentions
chunk=torch.nn.Module()
chunk.forward=MethodType(WhisperForConditionalGeneration.forward,chunk)
chunk._ispeak_retention_original=retention._ispeak_retention_original
assert install_active_chunk_release(chunk), 'Active-chunk optimization inactive'
assert any(h is _release_self_attentions for h in chunk._forward_hooks.values())
names='admission analysis_components assessment_details audio_transport_metadata bounded_dtw container_admission dependency_security filler_evidence filler_regression filler_hybrid input_validation memory_diagnostics model_integrity model_timestamps paired_evaluation readiness recording_duration reference_validation security_hardening timestamp_rejection transcript_diagnostics v6_candidate whisper_timestamp_memory whisper_longform_retention whisper_active_chunk whisper_fallback word_alignment'.split()
for flag in ['false','true']:
 os.environ['ISPEAK_MEMORY_DIAGNOSTICS']=flag
 with patch.object(model.ISpeakWhisper,'__init__',side_effect=AssertionError('Model loading forbidden')), patch.object(filler_words.LocalTokenFillerClassifier,'__init__',side_effect=AssertionError('Model loading forbidden')):
  suite=unittest.defaultTestLoader.loadTestsFromNames(['testing_script.test_'+n for n in names])
  result=unittest.TextTestRunner(verbosity=0).run(suite)
  assert result.wasSuccessful() and result.testsRun==189
 # Loader-contract tests invoke the real constructor with fully mocked factories.
 # Keep the original constructor prohibition above for every other suite, and
 # independently forbid all weight loading and pipeline creation here.
 with patch('transformers.WhisperProcessor.from_pretrained',side_effect=AssertionError('Model loading forbidden')), patch('transformers.WhisperForConditionalGeneration.from_pretrained',side_effect=AssertionError('Model loading forbidden')), patch('peft.PeftModel.from_pretrained',side_effect=AssertionError('Model loading forbidden')), patch('transformers.pipeline',side_effect=AssertionError('Model loading forbidden')), patch.object(filler_words.LocalTokenFillerClassifier,'__init__',side_effect=AssertionError('Model loading forbidden')):
  backend=unittest.defaultTestLoader.loadTestsFromName('testing_script.test_whisper_backend')
  backend_result=unittest.TextTestRunner(verbosity=0).run(backend)
  assert backend_result.wasSuccessful() and backend_result.testsRun==4
print(json.dumps({'model_free_tests_each_mode':193,'hook_active':True,'active_chunk_hook':True,'fallback_hook':True,'model_manifest_files':12,'inference':False}))
'''
    mount_args = []
    for source, target in [(tests / 'testing_script', '/app/testing_script'),
                           (tests / 'Dockerfile', '/app/Dockerfile'),
                           (tests / '.dockerignore', '/app/.dockerignore')]:
        mount_args += ['--mount', f'type=bind,source={source},target={target},readonly']
    # Runtime code/model layers remain read-only; only isolated scratch is writable.
    v.phase = 'model_free_image_checks'
    result = v.docker(*v.run_args('model-free-' + v.key, memory='4g'),
                     '--read-only', '--tmpfs', '/tmp/ispeak:rw,size=512m,mode=1777',
                     '-e', 'ISPEAK_ENV=development', '-e', 'ISPEAK_SERVICE_TOKEN=',
                     '-e', 'OMP_NUM_THREADS=2', '-e', 'MKL_NUM_THREADS=2',
                     *mount_args, '--entrypoint=python', v.image, '-B', '-c', code,
                     json.dumps(hashes), manifest['model_manifest_sha256'], seconds=600)
    summary = json.loads(result.decode().strip().splitlines()[-1])
    if summary != {'model_free_tests_each_mode': 193, 'hook_active': True, 'active_chunk_hook': True,
                   'fallback_hook': True, 'model_manifest_files': 12, 'inference': False}:
        raise RuntimeError('Incomplete model-free evidence')
    v.evidence('model_free_image', result='PASS', tests_each_mode=193,
               dependency_changes=sorted(targets), model_files=12, hook_active=True,
               active_chunk_hook=True, fallback_hook=True,
               inventory_sha256=hashlib.sha256(json.dumps(candidate, sort_keys=True).encode()).hexdigest(),
               full_model_inference=False)
