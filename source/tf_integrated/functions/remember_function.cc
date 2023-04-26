#include <array>
#include <iostream>
#include <string>
#include <type_traits>
#include <utility>

template <typename T, typename... Ts>
std::array<T, sizeof...(Ts)> toArray(Ts... x) {
  std::array<T, sizeof...(Ts)> arr{{(x)...}};
  return arr;
}

template <typename T, typename... Ts> void printAll(Ts... xs) {
  std::array<T, sizeof...(Ts)> arr = toArray<T>(xs...);
  for (auto &q : arr) {
    std::cout << q << ",";
  }
  std::cout << std::endl;
}

template <typename T, T... sametypes>
void print_sequence(std::integer_sequence<T, sametypes...> int_seq) {
  (void)(T[]){0, ((void)(std::cout << sametypes << ", "), 0)...};
  std::cout << '\n';
}

template <typename T, std::size_t... Ts, typename... Argtypes>
void call_function_with_helper(std::integer_sequence<std::size_t, Ts...>,
                               T (*function)(const Argtypes &...),
                               std::array<T, sizeof...(Ts)> arr) {
  std::cout << function(arr[Ts]...) << std::endl;
  printAll<T>(arr[Ts]...);
}

template <std::size_t I, typename T, typename... Argtypes>
void call_function_with(T (*function)(const Argtypes &...),
                        std::array<T, I> arr) {
  call_function_with_helper(std::make_index_sequence<sizeof...(Argtypes)>{},
                            function, arr);
}

template <typename T> T func(const T &x1, const T &x2) { return x1 + x2; }

int main() {

  // print_sequence(std::integer_sequence<unsigned, 9, 2, 5, 1, 9, 1, 6>{});
  // print_sequence(std::index_sequence_for<int, double, int>{});
  // print_sequence(std::make_index_sequence<4>{});

  // std::array<double, 3> a{{1.2, 1.4, 2.5}};
  std::array<double, 2> a{{1.2, 1.4}};
  call_function_with_helper(std::integer_sequence<std::size_t, 0, 1>{},
                            &func<double>, a);
  call_function_with_helper(std::make_index_sequence<a.size()>{}, &func<double>,
                            a);
  call_function_with_helper<double, 0, 1>(std::make_index_sequence<a.size()>{},
                                          &func<double>, a);
  call_function_with(&func<double>, a);

  return 0;
}