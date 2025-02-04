// Force H.264 baseline profile
const pc = new RTCPeerConnection({
  iceServers: [],
  encodedInsertableStreams: true,
  sdpSemantics: 'unified-plan',
  codecs: {
    video: [
      {
        mimeType: 'video/H264',
        clockRate: 90000,
        channels: 1,
        parameters: {
          'profile-level-id': '42e01f',
          'packetization-mode': 1,
          'level-asymmetry-allowed': 1
        }
      }
    ]
  }
}); 

pc.getTransceivers().forEach(t => {
  console.log('Codec parameters:', 
    t.sender.track.getSettings(),
    t.receiver.track.getSettings()
  );
});
// Should show profile-level-id=42e01f 