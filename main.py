import helpers

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.

def process_data(csv_file):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    train_data_array = helpers.get_data_from_csv('train.csv')
    print('train_data_array:\n', train_data_array, '\n')

    test_data_array = helpers.get_data_from_csv('test.csv')
    print('test_data_array:\n', test_data_array)

