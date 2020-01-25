#import <XCTest/XCTest.h>
#import <stdbool.h>

typedef struct
{
    size_t executed;
    size_t passed;
    bool runMain;
    bool summarize;
} UnitTestResult;

extern int rt_init(void);
extern int rt_term(void);
extern UnitTestResult runModuleUnitTests(void);

@interface TestRunnerTests : XCTestCase
@end

@implementation TestRunnerTests

- (void)tearDown
{
    XCTAssertTrue(rt_term());
}

- (void)test
{
    XCTAssertTrue(rt_init());
    UnitTestResult result = runModuleUnitTests();
    XCTAssertLessThanOrEqual(result.executed, result.passed);
}

@end
