
from collections import defaultdict

import importlib
import os
import sys
import traceback

# Choose a conda folder.
BASE_DIR = os.environ['CONDA_PREFIX']
site_packages_dir = os.path.join(BASE_DIR, 'lib/python3.8/site-packages')

package_registry = defaultdict(list)

for pkg in sorted(os.listdir(site_packages_dir)):
    if pkg.endswith(".dist-info"):
        pkg = pkg.replace(".dist-info", "")
        package_name, package_version = pkg.split("-")
        package_registry[package_name].append(package_version)
        
# Look for instances with multiple packages
for (package_name, package_dist_info_versions) in package_registry.items():
    if len(package_dist_info_versions) > 1:
        print(f"Multiple versions found for package {package_name}: {package_dist_info_versions}")
        
        # Try importing the package.
        try:
            package = importlib.__import__(package_name)
            correct_version = package.__version__
            print(f"Correct version: {correct_version}")
        except ModuleNotFoundError:
            print("Package does not actually exist.")
        except Exception as e:
            print("Error detecting package version.", file=sys.stderr)
            traceback.print_exc()
            continue
