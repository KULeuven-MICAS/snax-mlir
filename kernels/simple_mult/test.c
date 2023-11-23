extern int *test;
int main() {
  int test2 = 34;
  test = &test2;
  return 0;
}
