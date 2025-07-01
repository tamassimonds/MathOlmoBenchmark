#!/usr/bin/env python3
"""
Unit tests for math scoring functions
Tests the mathematical equivalence checking and answer extraction
"""

import unittest
from maths.reward_simple import (
    compute_score, 
    Score, 
    Status,
    last_boxed_only_string,
    remove_boxed,
    normalize_final_answer,
    is_equiv
)


class TestMathScoring(unittest.TestCase):
    
    def test_last_boxed_only_string(self):
        """Test extraction of boxed answers"""
        # Basic boxed answer
        self.assertEqual(
            last_boxed_only_string("The answer is \\boxed{42}"),
            "\\boxed{42}"
        )
        
        # Multiple boxed answers - should get the last one
        self.assertEqual(
            last_boxed_only_string("First \\boxed{10} then \\boxed{20}"),
            "\\boxed{20}"
        )
        
        # Nested braces
        self.assertEqual(
            last_boxed_only_string("\\boxed{{x^2} + 1}"),
            "\\boxed{{x^2} + 1}"
        )
        
        # No boxed answer
        self.assertIsNone(last_boxed_only_string("Just some text with no answer"))
        
        # fbox variant
        self.assertEqual(
            last_boxed_only_string("Using fbox: \\fbox{answer}"),
            "\\fbox{answer}"
        )
    
    def test_remove_boxed(self):
        """Test removal of boxed wrapper"""
        self.assertEqual(remove_boxed("\\boxed{42}"), "42")
        self.assertEqual(remove_boxed("\\boxed{x + y}"), "x + y")
        self.assertEqual(remove_boxed("\\boxed{\\frac{1}{2}}"), "\\frac{1}{2}")
        
        # Boxed with space
        self.assertEqual(remove_boxed("\\boxed 42"), "42")
    
    def test_normalize_final_answer(self):
        """Test answer normalization"""
        # Remove trailing period
        self.assertEqual(normalize_final_answer("42."), "42")
        
        # Remove dollar signs and commas
        self.assertEqual(normalize_final_answer("$1,234"), "1234")
        
        # Fraction to decimal
        self.assertEqual(normalize_final_answer("1/2"), "0.5")
        self.assertEqual(normalize_final_answer("3/4"), "0.75")
        
        # Handle division by zero gracefully
        self.assertEqual(normalize_final_answer("1/0"), "1/0")  # Should not crash
    
    def test_is_equiv(self):
        """Test mathematical equivalence checking"""
        # Exact matches
        self.assertTrue(is_equiv("42", "42"))
        self.assertTrue(is_equiv("x + y", "x + y"))
        
        # Numerical equivalence
        self.assertTrue(is_equiv("42", "42.0"))
        self.assertTrue(is_equiv("1/2", "0.5"))
        self.assertTrue(is_equiv("0.333333", "1/3"))  # Close enough
        
        # Case insensitive
        self.assertTrue(is_equiv("ABC", "abc"))
        
        # Whitespace handling
        self.assertTrue(is_equiv("x + y", "x+y"))
        self.assertTrue(is_equiv(" 42 ", "42"))
        
        # Different values
        self.assertFalse(is_equiv("42", "43"))
        self.assertFalse(is_equiv("x", "y"))
        
        # None handling
        self.assertFalse(is_equiv(None, "42"))
        self.assertFalse(is_equiv("42", None))
        self.assertFalse(is_equiv(None, None))
    
    def test_compute_score_boxed_correct(self):
        """Test scoring with correct boxed answers"""
        # Correct answer
        solution = "Let me solve this step by step. First... Therefore \\boxed{42}"
        ground_truth = "42"
        score = compute_score(solution, ground_truth)
        
        self.assertEqual(score.result, 1.0)
        self.assertEqual(score.status, Status.OK)
        self.assertEqual(score.report, solution)
    
    def test_compute_score_boxed_incorrect(self):
        """Test scoring with incorrect boxed answers"""
        solution = "Let me solve this step by step. First... Therefore \\boxed{43}"
        ground_truth = "42"
        score = compute_score(solution, ground_truth)
        
        self.assertEqual(score.result, 0.0)
        self.assertEqual(score.status, Status.NOT_EQUIV)
        self.assertEqual(score.report, solution)
    
    def test_compute_score_no_boxed(self):
        """Test scoring with no boxed answer"""
        solution = "I tried to solve this but got confused and gave up."
        ground_truth = "42"
        score = compute_score(solution, ground_truth)
        
        self.assertEqual(score.result, 0.0)
        self.assertEqual(score.status, Status.NO_INP)
        self.assertEqual(score.report, solution)
    
    def test_compute_score_pattern_matching(self):
        """Test scoring with common answer patterns (no boxed)"""
        test_cases = [
            "The answer is 42",
            "Therefore, 42",
            "So the final answer is 42.",
            "Thus, we get 42",
        ]
        
        for solution in test_cases:
            with self.subTest(solution=solution):
                score = compute_score(solution, "42")
                self.assertEqual(score.result, 1.0)
                self.assertEqual(score.status, Status.OK)
    
    def test_compute_score_numerical_equivalence(self):
        """Test scoring with numerically equivalent answers"""
        # Decimal vs fraction
        solution = "The calculation gives us \\boxed{0.5}"
        ground_truth = "1/2"
        score = compute_score(solution, ground_truth)
        self.assertEqual(score.result, 1.0)
        
        # Integer vs float
        solution = "The answer is \\boxed{42.0}"
        ground_truth = "42"
        score = compute_score(solution, ground_truth)
        self.assertEqual(score.result, 1.0)
    
    def test_compute_score_formatting_variations(self):
        """Test scoring with various formatting"""
        ground_truth = "42"
        
        test_cases = [
            "\\boxed{42}",
            "\\boxed{42.}",  # Trailing period
            "\\boxed{$42}",  # Dollar sign
            "\\boxed{ 42 }",  # Extra spaces
        ]
        
        for boxed_answer in test_cases:
            solution = f"After solving, we get {boxed_answer}"
            with self.subTest(boxed_answer=boxed_answer):
                score = compute_score(solution, ground_truth)
                self.assertEqual(score.result, 1.0)
    
    def test_compute_score_error_handling(self):
        """Test error handling in scoring"""
        # This should not crash even with malformed input
        solution = "\\boxed{malformed"  # Missing closing brace
        ground_truth = "42"
        score = compute_score(solution, ground_truth)
        
        # Should not crash and return some result
        self.assertIsInstance(score.result, (int, float))
        self.assertIsInstance(score.status, Status)
    
    def test_real_math_examples(self):
        """Test with realistic math problem solutions"""
        # Algebra problem
        solution1 = """
        Let's solve for x:
        2x + 3 = 7
        2x = 7 - 3
        2x = 4
        x = 2
        
        Therefore, \\boxed{2}
        """
        score1 = compute_score(solution1, "2")
        self.assertEqual(score1.result, 1.0)
        
        # Geometry problem  
        solution2 = """
        The area of a circle is πr².
        With radius 3: A = π(3)² = 9π
        
        The answer is \\boxed{9\\pi}
        """
        score2 = compute_score(solution2, "9π")
        self.assertEqual(score2.result, 1.0)
        
        # Fraction result
        solution3 = """
        Probability = favorable outcomes / total outcomes
        = 3/12 = 1/4
        
        \\boxed{\\frac{1}{4}}
        """
        score3 = compute_score(solution3, "1/4")
        self.assertEqual(score3.result, 1.0)


def run_tests():
    """Run all tests and return results"""
    print("🧮 Running Math Scoring Tests...")
    print("=" * 50)
    
    # Create a test suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestMathScoring)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Test Results:")
    print(f"✅ Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"❌ Failed: {len(result.failures)}")
    print(f"💥 Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback}")
    
    if result.errors:
        print("\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback}")
    
    print(f"\n🎯 Overall: {'PASS' if result.wasSuccessful() else 'FAIL'}")
    return result.wasSuccessful()


if __name__ == "__main__":
    run_tests()