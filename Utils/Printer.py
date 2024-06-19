from colorama import Fore


class Printer:

    def print_success(self: str):
        print(Fore.GREEN + self + Fore.RESET)

    def print_error(self: str):
        print(Fore.RED + self + Fore.RESET)

    def print_warning(self: str):
        print(Fore.YELLOW + self + Fore.RESET)

    def print_info(self: str):
        print(Fore.CYAN + self + Fore.RESET)