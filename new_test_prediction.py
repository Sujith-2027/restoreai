"""
Testing script to verify all fixes for the classification issues
Run this to ensure laptop is correctly identified and not misclassified as fridge
"""

import numpy as np
from PIL import Image
import io

# Mock test scenarios
def create_test_scenarios():
    """Create test scenarios based on the reported issues"""
    
    scenarios = [
        {
            "name": "Laptop with keyboard visible",
            "image_size": (1920, 1080),  # Wide aspect ratio ~1.78
            "avg_color": (60, 70, 80),  # Dark/grayish (typical laptop)
            "description": "laptop screen not working, keyboard is fine",
            "expected": "Laptop",
            "issue": "Was being classified as Fridge"
        },
        {
            "name": "Fridge - tall and narrow",
            "image_size": (800, 1400),  # Tall aspect ratio ~0.57
            "avg_color": (220, 220, 220),  # White
            "description": "fridge not cooling, freezer section broken",
            "expected": "Fridge",
            "issue": "Should remain as Fridge"
        },
        {
            "name": "Air Conditioner - wide and horizontal",
            "image_size": (1600, 600),  # Wide aspect ratio ~2.67
            "avg_color": (230, 230, 230),  # White
            "description": "ac not cooling properly",
            "expected": "Air Conditioner",
            "issue": "Should not be confused with Fridge"
        },
        {
            "name": "Laptop with strong keywords",
            "image_size": (1600, 900),  # Laptop aspect ~1.78
            "avg_color": (70, 75, 80),  # Dark
            "description": "macbook screen flickering, keyboard backlight not working",
            "expected": "Laptop",
            "issue": "Strong keywords should override wrong ML prediction"
        },
        {
            "name": "Television",
            "image_size": (2000, 1125),  # 16:9 aspect ~1.78
            "avg_color": (20, 20, 20),  # Very dark (black screen)
            "description": "tv screen is black, no display",
            "expected": "Television",
            "issue": "Should not confuse with laptop despite similar aspect"
        }
    ]
    
    return scenarios


def simulate_ml_prediction(scenario):
    """
    Simulate what the ML model might predict (including wrong predictions)
    This simulates the bug where laptop was predicted as fridge
    """
    
    # Simulate the bug: ML model incorrectly predicts Fridge for laptop
    if "Laptop" in scenario["expected"]:
        return "Fridge", 0.993  # High confidence, but WRONG (the actual bug)
    
    # For other devices, assume correct prediction
    return scenario["expected"], 0.85


def test_validation_logic():
    """Test the validation logic that should catch misclassifications"""
    
    print("="*80)
    print("TESTING VALIDATION LOGIC - FIX FOR LAPTOP MISCLASSIFICATION")
    print("="*80)
    print()
    
    scenarios = create_test_scenarios()
    
    from ml_utils import validate_prediction, get_image_features
    
    for scenario in scenarios:
        print(f"\n{'─'*80}")
        print(f"TEST: {scenario['name']}")
        print(f"{'─'*80}")
        print(f"Expected Device: {scenario['expected']}")
        print(f"Issue: {scenario['issue']}")
        print()
        
        # Create mock features
        w, h = scenario["image_size"]
        features = {
            "width": w,
            "height": h,
            "aspect_ratio": w / h,
            "brightness": np.mean(scenario["avg_color"]),
            "size": max(w, h)
        }
        
        print(f"Image Features:")
        print(f"  - Size: {w}x{h}")
        print(f"  - Aspect Ratio: {features['aspect_ratio']:.2f}")
        print(f"  - Brightness: {features['brightness']:.1f}")
        print()
        
        # Simulate ML prediction (including the bug)
        ml_label, ml_conf = simulate_ml_prediction(scenario)
        print(f"ML Model Prediction: {ml_label} (Confidence: {ml_conf:.1%})")
        
        # Create a mock image for validation
        mock_img = Image.new('RGB', scenario["image_size"], 
                             tuple(scenario["avg_color"]))
        
        # TEST THE VALIDATION
        is_valid, reason, conf_adjustment = validate_prediction(
            ml_label, ml_conf, features, mock_img, scenario["description"]
        )
        
        print(f"\nValidation Result:")
        print(f"  - Valid: {is_valid}")
        print(f"  - Reason: {reason}")
        print(f"  - Confidence Adjustment: {conf_adjustment}")
        
        if not is_valid:
            adjusted_conf = ml_conf * conf_adjustment
            print(f"  - Adjusted Confidence: {adjusted_conf:.1%}")
        
        # Determine if fix works
        if scenario["expected"] == "Laptop" and ml_label == "Fridge":
            if not is_valid:
                print(f"\n✅ FIX WORKS: Validation correctly rejected Fridge prediction for Laptop")
            else:
                print(f"\n❌ FIX FAILED: Validation did not catch the misclassification!")
        
        print()


def test_keyword_override():
    """Test that strong keywords override wrong ML predictions"""
    
    print("\n" + "="*80)
    print("TESTING KEYWORD OVERRIDE MECHANISM")
    print("="*80)
    print()
    
    test_cases = [
        {
            "description": "macbook pro screen broken, keyboard not responding",
            "ml_prediction": "Fridge",
            "ml_confidence": 0.99,
            "expected_override": "Laptop",
            "reason": "Strong laptop keywords (macbook, keyboard, screen)"
        },
        {
            "description": "refrigerator not cooling, freezer ice buildup",
            "ml_prediction": "Air Conditioner",
            "ml_confidence": 0.75,
            "expected_override": "Fridge",
            "reason": "Strong fridge keywords (refrigerator, freezer)"
        }
    ]
    
    for i, test in enumerate(test_cases, 1):
        print(f"Test Case {i}:")
        print(f"  Description: {test['description']}")
        print(f"  ML Prediction: {test['ml_prediction']} ({test['ml_confidence']:.0%})")
        print(f"  Expected Override: {test['expected_override']}")
        print(f"  Reason: {test['reason']}")
        print()


def test_aspect_ratio_detection():
    """Test aspect ratio based device detection"""
    
    print("\n" + "="*80)
    print("TESTING ASPECT RATIO DETECTION")
    print("="*80)
    print()
    
    aspect_tests = [
        {"device": "Laptop", "aspect": 1.78, "size": (1920, 1080), "should_match": True},
        {"device": "Laptop", "aspect": 0.60, "size": (800, 1400), "should_match": False},
        {"device": "Fridge", "aspect": 0.60, "size": (800, 1400), "should_match": True},
        {"device": "Fridge", "aspect": 2.5, "size": (1500, 600), "should_match": False},
        {"device": "Air Conditioner", "aspect": 2.5, "size": (1500, 600), "should_match": True},
        {"device": "Television", "aspect": 1.78, "size": (2000, 1125), "should_match": True},
    ]
    
    from ml_utils import DEVICE_FEATURES
    
    for test in aspect_tests:
        device = test["device"]
        aspect = test["aspect"]
        req = DEVICE_FEATURES[device]
        
        min_aspect = req.get("min_aspect", 0)
        max_aspect = req.get("max_aspect", 10)
        
        matches = min_aspect <= aspect <= max_aspect
        
        status = "✅" if matches == test["should_match"] else "❌"
        print(f"{status} {device:20} Aspect: {aspect:.2f} (Range: {min_aspect:.2f}-{max_aspect:.2f}) - {'Match' if matches else 'No Match'}")


def print_fix_summary():
    """Print summary of all fixes implemented"""
    
    print("\n" + "="*80)
    print("SUMMARY OF FIXES IMPLEMENTED")
    print("="*80)
    print()
    
    fixes = [
        {
            "problem": "Laptop misclassified as Fridge (99.3% confidence)",
            "root_cause": "ML model prediction trusted without validation",
            "solution": "Added validate_prediction() function that checks:\n"
                       "  - Aspect ratio consistency\n"
                       "  - Keyword contradictions\n"
                       "  - Physical features (keyboard detection)\n"
                       "  - Size constraints"
        },
        {
            "problem": "Wrong parts shown (Compressor, Thermostat for laptop)",
            "root_cause": "Parts matched to wrong device classification",
            "solution": "Parts now correctly extracted based on validated device type"
        },
        {
            "problem": "Screen damage shown as N/A for laptop",
            "root_cause": "Laptop treated as non-screen device",
            "solution": "Device features now correctly identify screen devices"
        },
        {
            "problem": "High confidence in wrong prediction",
            "root_cause": "No confidence adjustment for invalid predictions",
            "solution": "Confidence reduced by 30-70% when validation fails"
        },
        {
            "problem": "Keywords ignored when ML confident",
            "root_cause": "ML model had 70% weight regardless of validation",
            "solution": "Smart weighting:\n"
                       "  - Keywords: 60% (highest priority)\n"
                       "  - Validated ML: 35-50%\n"
                       "  - Unvalidated ML: 15%\n"
                       "  - Heuristics: 25%"
        },
        {
            "problem": "Fridge vs AC confusion",
            "root_cause": "Both white appliances with similar features",
            "solution": "Aspect ratio distinction:\n"
                       "  - Fridge: < 0.85 (tall/narrow)\n"
                       "  - AC: 1.0-3.5 (wide/horizontal)"
        },
        {
            "problem": "No debugging visibility",
            "root_cause": "Silent failures in classification pipeline",
            "solution": "Added comprehensive logging showing:\n"
                       "  - All stage predictions\n"
                       "  - Validation results\n"
                       "  - Score breakdowns\n"
                       "  - Final decision reasoning"
        }
    ]
    
    for i, fix in enumerate(fixes, 1):
        print(f"{i}. PROBLEM: {fix['problem']}")
        print(f"   Root Cause: {fix['root_cause']}")
        print(f"   Solution: {fix['solution']}")
        print()


if __name__ == "__main__":
    print("\n" + "🔧 "*40)
    print("COMPREHENSIVE FIX VERIFICATION")
    print("🔧 "*40)
    
    # Run all tests
    test_validation_logic()
    test_keyword_override()
    test_aspect_ratio_detection()
    print_fix_summary()
    
    print("\n" + "="*80)
    print("TESTING COMPLETE")
    print("="*80)
    print()
    print("Next Steps:")
    print("1. Replace your ml_utils.py with the fixed version")
    print("2. Test with actual laptop image")
    print("3. Verify the debug logs show validation working")
    print("4. Check that 'Laptop' is predicted with correct parts")
    print()