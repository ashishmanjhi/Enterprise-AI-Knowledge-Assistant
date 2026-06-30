"""
Test runner script for Enterprise RAG Platform
Runs different test suites and generates reports
"""
import sys
import subprocess
import argparse
from pathlib import Path
from datetime import datetime


def run_command(cmd, description):
    """Run a command and handle output"""
    print(f"\n{'='*70}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            check=False
        )
        return result.returncode == 0
    except Exception as e:
        print(f"Error running command: {e}")
        return False


def run_unit_tests(verbose=False):
    """Run unit tests"""
    cmd = ["pytest", "backend/tests/test_loaders.py", "backend/tests/test_chunking.py"]
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    cmd.extend(["-m", "not integration and not e2e"])
    
    return run_command(cmd, "Unit Tests")


def run_integration_tests(verbose=False):
    """Run integration tests"""
    cmd = ["pytest", "backend/tests/test_api_integration.py"]
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    return run_command(cmd, "Integration Tests")


def run_all_tests(verbose=False):
    """Run all tests"""
    cmd = ["pytest", "backend/tests/"]
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    return run_command(cmd, "All Tests")


def run_with_coverage():
    """Run tests with coverage report"""
    cmd = [
        "pytest",
        "backend/tests/",
        "--cov=backend",
        "--cov-report=html",
        "--cov-report=term-missing",
        "-v"
    ]
    
    success = run_command(cmd, "Tests with Coverage")
    
    if success:
        print("\n" + "="*70)
        print("Coverage report generated in htmlcov/index.html")
        print("="*70)
    
    return success


def run_specific_test(test_path, verbose=False):
    """Run a specific test file or test"""
    cmd = ["pytest", test_path]
    if verbose:
        cmd.append("-vv")
    else:
        cmd.append("-v")
    
    return run_command(cmd, f"Specific Test: {test_path}")


def check_dependencies():
    """Check if required dependencies are installed"""
    print("\nChecking dependencies...")
    
    required = ["pytest", "pytest-asyncio"]
    missing = []
    
    for package in required:
        try:
            __import__(package.replace("-", "_"))
            print(f"[OK] {package} is installed")
        except ImportError:
            print(f"[FAIL] {package} is NOT installed")
            missing.append(package)
    
    if missing:
        print(f"\nMissing dependencies: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        return False
    
    print("\n[OK] All dependencies are installed")
    return True


def generate_test_report():
    """Generate a test report"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"test_report_{timestamp}.txt"
    
    cmd = [
        "pytest",
        "backend/tests/",
        "-v",
        "--tb=short",
        f"--html=test_report_{timestamp}.html",
        "--self-contained-html"
    ]
    
    success = run_command(cmd, "Generate Test Report")
    
    if success:
        print(f"\n[OK] Test report generated: test_report_{timestamp}.html")
    
    return success


def main():
    """Main test runner"""
    parser = argparse.ArgumentParser(
        description="Test runner for Enterprise RAG Platform"
    )
    
    parser.add_argument(
        "suite",
        nargs="?",
        choices=["unit", "integration", "all", "coverage", "report"],
        default="all",
        help="Test suite to run (default: all)"
    )
    
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "-t", "--test",
        type=str,
        help="Run specific test file or test"
    )
    
    parser.add_argument(
        "--check-deps",
        action="store_true",
        help="Check if dependencies are installed"
    )
    
    args = parser.parse_args()
    
    # Check dependencies first
    if args.check_deps or not check_dependencies():
        if not check_dependencies():
            sys.exit(1)
        return
    
    print("\n" + "="*70)
    print("Enterprise RAG Platform - Test Runner")
    print("="*70)
    
    # Run specific test if provided
    if args.test:
        success = run_specific_test(args.test, args.verbose)
        sys.exit(0 if success else 1)
    
    # Run selected test suite
    if args.suite == "unit":
        success = run_unit_tests(args.verbose)
    elif args.suite == "integration":
        success = run_integration_tests(args.verbose)
    elif args.suite == "coverage":
        success = run_with_coverage()
    elif args.suite == "report":
        success = generate_test_report()
    else:  # all
        success = run_all_tests(args.verbose)
    
    # Print summary
    print("\n" + "="*70)
    if success:
        print("[SUCCESS] Tests completed successfully!")
    else:
        print("[FAILED] Some tests failed. Check output above for details.")
    print("="*70 + "\n")
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

# Made with Bob
