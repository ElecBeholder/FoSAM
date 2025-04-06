# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    import torch

    file_path = "/home/lwx/test/DynamicFocus/a_records_train/params_mask_segformer.pt" 
    data = torch.load(file_path)

    for key, value in data.items():
        print(f"Key: {key}, Value Type: {type(value)}")



# See PyCharm help at https://www.jetbrains.com/help/pycharm/
