import os
import fnmatch

def find_trajectory_paths(path=".", search_pattern = "^$", exclude_patterns = []):
    runner_files = []
    
    for root, dirnames, filenames in os.walk(path):
        for filename in fnmatch.filter(filenames, search_pattern):
            runner_path = os.path.join(root, filename)
            runner_path_excluded = False
            for exclude_pattern in exclude_patterns:
                regex_pattern = re.compile(exclude_pattern)
                match = regex_pattern.search(runner_path)
                
                if match:
                    runner_path_excluded = True
            if runner_path_excluded == False:
                runner_files.append(runner_path)
                
    return runner_files