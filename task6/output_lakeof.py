import os

def find_missing_numbers(directory):
    file_list = os.listdir(directory)
    existing_numbers = set([int(filename.split('.')[0]) for filename in file_list if filename.endswith('.png')])

    max_number = max(existing_numbers)
    all_numbers = set(range(max_number + 1))

    missing_numbers = list(all_numbers - existing_numbers)
    missing_numbers.sort()

    return missing_numbers

directory = r'E:\桌面\自己的direct2global\testB'
missing_numbers = find_missing_numbers(directory)

if len(missing_numbers) == 0:
    print("No missing numbers found.")
else:
    print("Missing numbers:")
    for i, number in enumerate(missing_numbers, start=1):
        print(number, end='\n' if i % 10 == 0 else ' ')
