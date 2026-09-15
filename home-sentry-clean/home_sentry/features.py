"""Shared packet feature schema used by training and live inference."""

STATIC_FEATURES = [
    'is_broadcast', 'is_to_ap', 'is_from_ap', 'packet_frequency', 'window_time_delta',
    'frame.len', 'frame.cap_len', 'frame.time_delta', 'frame.time_relative', 'wlan.duration',
    'radiotap.length', 'radiotap.present.tsft', 'radiotap.present.flags', 'radiotap.present.rate',
    'radiotap.present.channel', 'radiotap.present.fhss', 'radiotap.present.dbm_antsignal',
    'radiotap.present.dbm_antnoise', 'radiotap.present.lock_quality', 'radiotap.present.tx_attenuation',
    'radiotap.present.db_tx_attenuation', 'radiotap.dbm_antsignal', 'radiotap.datarate', 'radiotap.channel.freq',
    'wlan.fc.type', 'wlan.fc.subtype', 'wlan.fc.version', 'wlan.fc.tods', 'wlan.fc.fromds',
    'wlan.fc.frag', 'wlan.fc.retry', 'wlan.fc.pwrmgt', 'wlan.fc.moredata', 'wlan.fc.protected', 'wlan.fc.order',
    'wlan.frag', 'wlan.seq', 'wlan.ba.control.ackpolicy',
    'wlan.qos.tid', 'wlan.qos.priority', 'wlan.qos.eosp', 'wlan.qos.ack', 'wlan.qos.amsdupresent',
    'wlan.qos.buf_state_indicated', 'wlan.qos.bit4',
    'wlan.fixed.capabilities.ess', 'wlan.fixed.capabilities.ibss', 'wlan.fixed.capabilities.privacy',
    'wlan.fixed.capabilities.spec_man', 'wlan.fixed.capabilities.short_slot_time', 'wlan.fixed.reason_code',
    'wlan.fixed.status_code', 'wlan.fixed.timestamp',
    'wlan.fixed.auth.alg', 'wlan.fixed.auth_seq', 'eapol.type', 'eap.code', 'tls.record.version'
]

ENGINEERED_FEATURES = ['is_broadcast','is_to_ap','is_from_ap','packet_frequency','window_time_delta']
CORE_TSHARK_FIELDS = [f for f in STATIC_FEATURES if f not in ENGINEERED_FEATURES] + ['wlan.ra','wlan.ta']
SEQUENCE_LENGTH = 20
