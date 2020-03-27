#import <XCTest/XCTest.h>
#include <stdio.h>

@interface bitrise_testTests : XCTestCase
{
    NSURL* url;
}
@end

@implementation bitrise_testTests

- (void)setUp {
    url = [NSURL fileURLWithPath:[NSTemporaryDirectory() stringByAppendingPathComponent:@"foo.txt"]];
    errno = 0;
    FILE* result = freopen(url.path.UTF8String, "w+", stdout);
//    XCTAssertEqualObjects(url.path, @"asd");
    XCTAssertEqual(errno, 0);
    XCTAssertNotEqual(result, NULL);
}

- (void)tearDown {
    freopen("/dev/tty", "w", stdout);

    XCTAttachment* attachment = [XCTAttachment attachmentWithContentsOfFileAtURL:url];
    attachment.lifetime = XCTAttachmentLifetimeKeepAlways;
    [self addAttachment:attachment];
}

- (void)testAsd {
#ifdef __arm64__
    printf("************************* 3 ARM64\n");
    XCTAssert(true, "ARM64");
#elif __x86_64__
    printf("************************* 3 x86-64\n");
    XCTAssert(true, "x86-64");
#endif
}
@end
