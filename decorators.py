def test_verbose(func):
    def wrapper(*args):
        print("="*50)
        func(*args)
        print(f"Test {func.__name__} is performed")
        print("="*50)
    return wrapper

