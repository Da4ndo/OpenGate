"""Test the OpenGate Detection integration files and structure."""
import json
import os


def test_manifest_exists_and_valid():
    """Test that manifest.json exists and is properly formatted."""
    manifest_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'custom_components', 
        'opengate_detection', 
        'manifest.json'
    )
    
    assert os.path.exists(manifest_path), "manifest.json file is missing"
    
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)
    
    # Check required fields
    required_fields = ['domain', 'name', 'version', 'requirements', 'codeowners']
    for field in required_fields:
        assert field in manifest, f"Missing required field: {field}"
    
    assert manifest['domain'] == 'opengate_detection'
    assert manifest['name'] == 'OpenGate Detection'
    assert manifest['version'] == '1.0.2'
    assert isinstance(manifest['requirements'], list)
    assert len(manifest['requirements']) > 0


def test_hacs_json_valid():
    """Test that hacs.json is properly formatted."""
    hacs_path = os.path.join(os.path.dirname(__file__), '..', 'hacs.json')
    
    assert os.path.exists(hacs_path), "hacs.json file is missing"
    
    with open(hacs_path, 'r') as f:
        hacs_config = json.load(f)
    
    # Check required fields for HACS
    required_fields = ['name', 'render_readme']
    for field in required_fields:
        assert field in hacs_config, f"Missing required field: {field}"
    
    assert hacs_config['name'] == 'OpenGate Detection'
    assert isinstance(hacs_config['render_readme'], bool)


def test_integration_structure():
    """Test that the integration has the required file structure."""
    base_path = os.path.join(os.path.dirname(__file__), '..', 'custom_components', 'opengate_detection')
    
    required_files = [
        '__init__.py',
        'manifest.json',
        'const.py',
        'config_flow.py',
        'binary_sensor.py',
        'sensor.py',
        'gate_detector.py',
        'services.yaml'
    ]
    
    for file in required_files:
        file_path = os.path.join(base_path, file)
        assert os.path.exists(file_path), f"Required file missing: {file}"


def test_translations_exist():
    """Test that translation files exist."""
    translations_path = os.path.join(
        os.path.dirname(__file__), 
        '..', 
        'custom_components', 
        'opengate_detection', 
        'translations'
    )
    
    assert os.path.exists(translations_path), "translations directory is missing"
    
    en_path = os.path.join(translations_path, 'en.json')
    assert os.path.exists(en_path), "English translation file is missing"
    
    with open(en_path, 'r') as f:
        translations = json.load(f)
    
    # Check that translations have required sections
    assert 'config' in translations, "Missing config translations"


def test_requirements_format():
    """Test that requirements.txt is properly formatted."""
    requirements_path = os.path.join(os.path.dirname(__file__), '..', 'requirements.txt')
    
    assert os.path.exists(requirements_path), "requirements.txt file is missing"
    
    with open(requirements_path, 'r') as f:
        requirements = f.read().strip().split('\n')
    
    # Should have OpenCV and other dependencies
    opencv_found = any('opencv' in req.lower() for req in requirements)
    assert opencv_found, "OpenCV dependency not found in requirements.txt" 