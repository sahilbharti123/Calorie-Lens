#import <React/RCTBridgeModule.h>
#import <React/RCTEventEmitter.h>

@interface RCT_EXTERN_MODULE(VigorlyWorkoutBridge, RCTEventEmitter)
RCT_EXTERN_METHOD(syncWorkout:(NSDictionary *)payload)
RCT_EXTERN_METHOD(sendCommand:(NSString *)command)
RCT_EXTERN_METHOD(drainEvents:(RCTPromiseResolveBlock)resolve rejecter:(RCTPromiseRejectBlock)reject)
@end
