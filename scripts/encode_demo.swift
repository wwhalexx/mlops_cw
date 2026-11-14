// macOS-only: encode actual command-output slides as a 90-second H.264 MP4.
// swift scripts/encode_demo.swift dist/demo dist/demo.mp4
import AppKit
import AVFoundation

let folder = URL(fileURLWithPath: CommandLine.arguments[1])
let output = URL(fileURLWithPath: CommandLine.arguments[2])
let files = try FileManager.default.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil)
    .filter { $0.pathExtension == "png" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
guard files.count == 9 else { fatalError("Expected nine verified command slides") }
let writer = try AVAssetWriter(outputURL: output, fileType: .mp4)
let input = AVAssetWriterInput(mediaType: .video, outputSettings: [
    AVVideoCodecKey: AVVideoCodecType.h264,
    AVVideoWidthKey: 1600, AVVideoHeightKey: 1000,
    AVVideoCompressionPropertiesKey: [AVVideoAverageBitRateKey: 2_500_000]
])
let adaptor = AVAssetWriterInputPixelBufferAdaptor(assetWriterInput: input,
    sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB,
                                 kCVPixelBufferWidthKey as String: 1600,
                                 kCVPixelBufferHeightKey as String: 1000])
writer.add(input)
guard writer.startWriting() else { fatalError("Cannot start encoder: \(String(describing: writer.error))") }
writer.startSession(atSourceTime: .zero)
for (slide, file) in files.enumerated() {
    let image = NSImage(contentsOf: file)!
    var rect = NSRect(x: 0, y: 0, width: 1600, height: 1000)
    let cgImage = image.cgImage(forProposedRect: &rect, context: nil, hints: nil)!
    var buffer: CVPixelBuffer?
    CVPixelBufferPoolCreatePixelBuffer(nil, adaptor.pixelBufferPool!, &buffer)
    CVPixelBufferLockBaseAddress(buffer!, [])
    let context = CGContext(data: CVPixelBufferGetBaseAddress(buffer!), width: 1600, height: 1000,
        bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer!),
        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
    context.draw(cgImage, in: CGRect(x: 0, y: 0, width: 1600, height: 1000))
    CVPixelBufferUnlockBaseAddress(buffer!, [])
    for frame in 0..<20 {
        while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.01) }
        guard adaptor.append(buffer!, withPresentationTime: CMTime(value: Int64(slide * 20 + frame),
                                                                   timescale: 2)) else {
            fatalError("Append failed: \(String(describing: writer.error))")
        }
    }
}
writer.endSession(atSourceTime: CMTime(seconds: 90, preferredTimescale: 2))
input.markAsFinished()
let semaphore = DispatchSemaphore(value: 0)
writer.finishWriting { semaphore.signal() }
semaphore.wait()
guard writer.status == .completed else { fatalError("Encode failed: \(String(describing: writer.error))") }
print(output.path)
