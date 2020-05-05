from __future__ import print_function

import numpy as np


def printMatrix(arr):
    for i in range(9):
        for j in range(9):
            print(arr[i][j], end=" ")
        print()


def find_empty_location(arr, l):
    for row in range(9):
        for col in range(9):
            if arr[row][col] == 0:
                l[0] = row
                l[1] = col
                return True
    return False


def used_in_col(arr, col, num):
    for i in range(9):
        if arr[i][col] == num:
            return True
    return False


def used_in_row(arr, row, num):
    for i in range(9):
        if arr[row][i] == num:
            return True
    return False


def used_in_box(arr, row, col, num):
    for i in range(3):
        for j in range(3):
            if arr[row + i][col + j] == num:
                return True
    return False


def check_if_can_be_placed(arr, row, col, num):
    flag = 0
    if used_in_col(arr, col, num):
        flag = 1
    if used_in_row(arr, row, num):
        flag = 1
    if used_in_box(arr, row - row % 3, col - col % 3, num):
        flag = 1
    if flag == 1:
        return False
    return True


def solve_sudoku(arr):

    l = [0, 0]

    if find_empty_location(arr, l) == False:
        return True

    # we got a row and column to fill from the above function if it returned True
    row = l[0]
    col = l[1]

    for num in range(1, 10):
        if check_if_can_be_placed(arr, row, col, num):

            # tentatively place
            arr[row][col] = num

            if solve_sudoku(arr) == True:
                return True

            # else failure
            arr[row][col] = 0

    # trigger backtracking
    return False
