"""
# DO NOT MODIFY THIS FILE.
#
# Any modifications to this file will not be used when reviewing your Submission.
"""
from Solution import Solution


def save(output, filename='Results.txt'):
    with open(filename, 'a') as file:
        file.write(output + '\n')


def start():
    filename = 'Results.txt'
    open(filename, 'w').close()


def execute():
    problem_sets = []

    with open(r'D:\tanishkaka\SlotsProject\ProblemSetList.txt', 'r') as file:
        for line in file:
            problem_sets.append(line.strip())

    for problem in problem_sets:
        solution = Solution()
        result = solution.get_answer(problem)
        if len(result) == 2:
            save(' , '.join(map(str, result)))
