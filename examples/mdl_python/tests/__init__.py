import unittest
import os

report_coverage: bool = os.getenv('CREATE_COVERAGE_REPORT') == "ON"

if report_coverage:
    try:
        import coverage
    except Exception:
        print("Unittest Coverage can not be recorded. The 'coverage' python module needs to be installed.")
        report_coverage = False


def run() -> int:
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern="test_*.py")
    runner = unittest.TextTestRunner()
    result: unittest.TestResult = runner.run(suite)
    if len(result.errors) == 0 and len(result.failures) == 0:
        return 0
    else:
        print(f"error : unit tests exited with {len(result.errors)} errors and {len(result.failures)} failures.")
        return -1
    
# run all tests
# independent of the cmake option we generate a report here in case the coverage module is installed
if __name__ == '__main__':
    ret_code: int = -2
    if report_coverage:
        folder: str = 'coverage_report'
        cov = coverage.Coverage(data_file=folder + '/.coverage')
        cov.start()
        ret_code = run()
        cov.stop()
        cov.save()
        cov.html_report(directory=folder, omit="**/__init__.py")
    else:
        ret_code = run()
    exit(ret_code)
