import unittest
import os

report_coverage: bool = True

if report_coverage:
    try:
        import coverage
    except Exception:
        print("Unittest Coverage can not be recorded. The 'coverage' python module needs to be installed.")
        report_coverage = False


def run():
    loader = unittest.TestLoader()
    suite = loader.discover(os.path.dirname(os.path.abspath(__file__)), pattern="test_*.py")
    runner = unittest.TextTestRunner()
    runner.run(suite)
    print("Done: Running Tests")


# run all tests
# independent of the cmake option we generate a report here in case the coverage module is installed
if __name__ == '__main__':
    if report_coverage:
        folder: str = 'coverage_report'
        cov = coverage.Coverage(data_file=folder + '/.coverage')
        cov.exclude('.*No constructor defined - class is abstract.*')
        cov.start()
        run()
        cov.stop()
        cov.save()
        cov.html_report(directory=folder)
        print("Done: Creating Coverage Report")
    else:
        run()
