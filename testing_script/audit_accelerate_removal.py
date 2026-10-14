"""Expected-failure probe for the disposable Accelerate-removal image only."""
import importlib.metadata
import json
import traceback


def main():
    report = {}
    try:
        importlib.metadata.version('accelerate')
    except importlib.metadata.PackageNotFoundError:
        report['accelerate_removed'] = True
    else:
        raise AssertionError('Run only in the disposable removal image')
    try:
        from peft import PeftModel  # noqa: F401
    except ModuleNotFoundError as exc:
        report['peft_import_failure'] = {'category': type(exc).__name__, 'missing_module': exc.name,
            'chain': [{'file': frame.filename.replace('\\', '/').split('/')[-1], 'function': frame.name}
                for frame in traceback.extract_tb(exc.__traceback__)]}
        assert exc.name == 'accelerate'
    else:
        raise AssertionError('PEFT unexpectedly imports without Accelerate')
    from fastapi.testclient import TestClient
    import fastapi_backend as backend
    with TestClient(backend.app) as client:
        health = client.get('/health')
        assert health.status_code == 503 and backend.app.state.model is None
        response = client.post('/transcribe', headers={'X-iSpeak-Service-Token': backend.SERVICE_TOKEN})
        assert response.status_code == 503
        report.update({'health_status': health.status_code, 'speech_model_ready': False,
                       'filler_ready': backend.app.state.filler_ready, 'analysis_status': response.status_code,
                       'removal_safe': False})
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
