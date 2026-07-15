#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <HardwareSerial.h>
#include <WiFi.h>
#include "esp_bt.h"
#include "mavlink.h"

// =========================== CONFIG =======================================
#define I2C_SDA               21
#define I2C_SCL               22
#define I2C_FREQ              100000UL

#define SD_CS                 5
#define SD_LOG_PATH           "/atlas_waypoints.csv"

#define MAV_UART_NUM          2
#define MAV_TX_PIN            17
#define MAV_RX_PIN            16
#define MAV_BAUD              57600
#define MAV_SYSID             42
#define MAV_COMPID            MAV_COMP_ID_PERIPHERAL

#define EZO_ADDR_PH           0x63
#define EZO_ADDR_EC           0x64
#define EZO_ADDR_RTD          0x66
#define EZO_ADDR_DO           0x61

// Measurement timing
#define STABILIZE_MS          60000UL          // probes settle
#define CAPTURE_MS            30000UL          // averaging window
#define READ_INTERVAL_MS      2000UL           // target sensor poll cadence
#define MAX_AWAKE_MS          180000UL         // hard ceiling per cycle

#define EZO_READ_DELAY_MS     900
#define EZO_RETRY_DELAY_MS    300              // extra wait on status 254
#define EZO_CMD_DELAY_MS      300

// Only push new T to pH/EC/DO when temperature drifts more than this
#define TEMP_DEADBAND_C       0.2f

// Idle loop pause - 50ms is 20x faster than PX4's 1Hz EXTENDED_SYS_STATE
#define IDLE_POLL_MS          50
// =========================================================================

RTC_DATA_ATTR uint32_t waypointCount = 0;

HardwareSerial MavSerial(MAV_UART_NUM);
bool sdOk = false;

// State tracking
uint8_t  lastLandedState   = MAV_LANDED_STATE_UNDEFINED;
uint8_t  currLandedState   = MAV_LANDED_STATE_UNDEFINED;
uint32_t lastHeartbeatRx   = 0;

// Cached temperature last pushed to each compensating probe.
// Index: 0=pH, 1=EC, 2=DO. Reset to NAN at start of each cycle.
float lastTempSent[3] = {NAN, NAN, NAN};

// ============== EZO I2C primitives ==============
bool ezoSendCommand(uint8_t addr, const char *cmd) {
  Wire.beginTransmission(addr);
  Wire.write((const uint8_t*)cmd, strlen(cmd));
  return (Wire.endTransmission() == 0);
}

// Returns the EZO status byte (1=ok, 2=syntax err, 254=processing, 255=nodata)
// or 0 if the bus didn't even answer.
uint8_t ezoReadResponse(uint8_t addr, char *outBuf, size_t bufLen, uint32_t waitMs) {
  delay(waitMs);
  Wire.requestFrom((int)addr, 32);
  if (!Wire.available()) return 0;
  uint8_t status = Wire.read();
  size_t i = 0;
  while (Wire.available() && i < bufLen - 1) {
    char c = (char)Wire.read();
    if (c == 0) break;
    outBuf[i++] = c;
  }
  outBuf[i] = '\0';
  return status;
}

float ezoReadFloat(uint8_t addr) {
  if (!ezoSendCommand(addr, "R")) return NAN;
  char buf[32] = {0};
  uint8_t status = ezoReadResponse(addr, buf, sizeof(buf), EZO_READ_DELAY_MS);
  if (status == 1) return atof(buf);

  // 254 = "still processing" - retry once without resending R.
  if (status == 254) {
    status = ezoReadResponse(addr, buf, sizeof(buf), EZO_RETRY_DELAY_MS);
    if (status == 1) return atof(buf);
  }
  return NAN;
}

void ezoSetTemp(uint8_t addr, float tempC) {
  if (isnan(tempC)) return;
  char cmd[16];
  snprintf(cmd, sizeof(cmd), "T,%.2f", tempC);
  ezoSendCommand(addr, cmd);
  delay(EZO_CMD_DELAY_MS);
  Wire.requestFrom((int)addr, 32);
  while (Wire.available()) Wire.read();
}

// Only push T if the new value differs from the last sent by more than the
// deadband. Saves ~900ms per sample loop when temperature is stable.
void ezoMaybeSetTemp(int idx, uint8_t addr, float tempC) {
  if (isnan(tempC)) return;
  if (!isnan(lastTempSent[idx]) &&
      fabsf(tempC - lastTempSent[idx]) < TEMP_DEADBAND_C) return;
  ezoSetTemp(addr, tempC);
  lastTempSent[idx] = tempC;
}

// ============== MAVLink tx helpers ==============
void mavSendNamedFloat(const char *name, float value) {
  mavlink_message_t msg;
  uint8_t buf[MAVLINK_MAX_PACKET_LEN];
  mavlink_msg_named_value_float_pack(
      MAV_SYSID, MAV_COMPID, &msg,
      (uint32_t)millis(), name, value);
  uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
  MavSerial.write(buf, len);
}

void mavSendHeartbeat() {
  mavlink_message_t msg;
  uint8_t buf[MAVLINK_MAX_PACKET_LEN];
  mavlink_msg_heartbeat_pack(
      MAV_SYSID, MAV_COMPID, &msg,
      MAV_TYPE_ONBOARD_CONTROLLER, MAV_AUTOPILOT_INVALID,
      0, 0, MAV_STATE_ACTIVE);
  uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
  MavSerial.write(buf, len);
}

void mavSendStatusText(const char *txt) {
  mavlink_message_t msg;
  uint8_t buf[MAVLINK_MAX_PACKET_LEN];
  mavlink_msg_statustext_pack(MAV_SYSID, MAV_COMPID, &msg,
      MAV_SEVERITY_INFO, txt, 0, 0);
  uint16_t len = mavlink_msg_to_send_buffer(buf, &msg);
  MavSerial.write(buf, len);
}

// ============== MAVLink rx parsing ==============
void pollMavlink() {
  mavlink_message_t msg;
  mavlink_status_t  status;

  while (MavSerial.available() > 0) {
    uint8_t c = MavSerial.read();
    if (mavlink_parse_char(MAVLINK_COMM_0, c, &msg, &status)) {
      switch (msg.msgid) {

        case MAVLINK_MSG_ID_HEARTBEAT:
          lastHeartbeatRx = millis();
          break;

        case MAVLINK_MSG_ID_EXTENDED_SYS_STATE: {
          mavlink_extended_sys_state_t ess;
          mavlink_msg_extended_sys_state_decode(&msg, &ess);
          currLandedState = ess.landed_state;
          break;
        }

        default:
          break;
      }
    }
  }
}

// ============== SD ==============
void sdInit() {
  if (SD.begin(SD_CS)) {
    sdOk = true;
    if (!SD.exists(SD_LOG_PATH)) {
      File f = SD.open(SD_LOG_PATH, FILE_WRITE);
      if (f) {
        f.println("waypoint,millis,n_samples,"
                  "temp_c,ph,ec_us_cm,do_mg_l,"
                  "temp_std,ph_std,ec_std,do_std,flags");
        f.close();
      }
    }
  }
}

void sdLogSample(uint32_t wp, uint32_t ts, uint16_t n,
                 float t, float ph, float ec, float doMgL,
                 float tStd, float phStd, float ecStd, float doStd,
                 const char *flags) {
  if (!sdOk) return;
  File f = SD.open(SD_LOG_PATH, FILE_APPEND);
  if (!f) return;
  f.printf("%lu,%lu,%u,%.3f,%.3f,%.1f,%.3f,%.3f,%.3f,%.2f,%.3f,%s\n",
           wp, ts, n, t, ph, ec, doMgL, tStd, phStd, ecStd, doStd, flags);
  f.close();
}

// ============== Sample averaging ==============
struct Accum {
  double sum = 0, sumSq = 0;
  uint16_t n = 0;
  uint16_t nan_count = 0;
  void add(float v) {
    if (isnan(v)) { nan_count++; return; }
    sum += v; sumSq += (double)v * v; n++;
  }
  float mean() const { return n ? (float)(sum / n) : NAN; }
  float stdev() const {
    if (n < 2) return 0.0f;
    double m = sum / n;
    double var = (sumSq / n) - (m * m);
    if (var < 0) var = 0;
    return (float)sqrt(var);
  }
};

// ============== Measurement state machine ==============
enum Phase { PHASE_STABILIZE, PHASE_CAPTURE, PHASE_DONE };

void runMeasurementCycle() {
  waypointCount++;

  // Reset cached temperatures so first sample always pushes T to each probe
  lastTempSent[0] = lastTempSent[1] = lastTempSent[2] = NAN;

  // Make sure EZOs are in polled mode (harmless if already set)
  const uint8_t addrs[] = {EZO_ADDR_PH, EZO_ADDR_EC, EZO_ADDR_RTD, EZO_ADDR_DO};
  for (uint8_t a : addrs) {
    ezoSendCommand(a, "C,0");
    delay(EZO_CMD_DELAY_MS);
    Wire.requestFrom((int)a, 32);
    while (Wire.available()) Wire.read();
  }

  char msg[50];   // MAVLink STATUSTEXT payload is exactly 50 bytes
  snprintf(msg, sizeof(msg), "ATLAS: WP %lu start", waypointCount);
  mavSendStatusText(msg);

  Accum aT, aPh, aEc, aDo;
  Phase phase = PHASE_STABILIZE;

  const uint32_t t0 = millis();
  uint32_t lastRead = 0;
  uint32_t lastHb   = 0;

  while (phase != PHASE_DONE) {
    uint32_t now = millis();

    // Safety: hard ceiling
    if (now - t0 > MAX_AWAKE_MS) {
      mavSendStatusText("ATLAS: max-awake timeout");
      break;
    }

    // Safety: if drone took off again mid-cycle, abort
    pollMavlink();
    if (currLandedState == MAV_LANDED_STATE_IN_AIR ||
        currLandedState == MAV_LANDED_STATE_TAKEOFF) {
      mavSendStatusText("ATLAS: airborne mid-cycle, aborting");
      break;
    }

    // 1 Hz heartbeat
    if (now - lastHb >= 1000) { lastHb = now; mavSendHeartbeat(); }

    // Phase transitions
    uint32_t elapsed = now - t0;
    if (phase == PHASE_STABILIZE && elapsed >= STABILIZE_MS) {
      phase = PHASE_CAPTURE;
      mavSendStatusText("ATLAS: capture phase");
    }
    if (phase == PHASE_CAPTURE && elapsed >= STABILIZE_MS + CAPTURE_MS) {
      phase = PHASE_DONE;
      break;
    }

    // Read cadence
    if (now - lastRead < READ_INTERVAL_MS) { delay(20); continue; }
    lastRead = now;

    // RTD first -> temp comp (only if changed) -> rest
    float tempC = ezoReadFloat(EZO_ADDR_RTD);
    if (!isnan(tempC)) {
      ezoMaybeSetTemp(0, EZO_ADDR_PH, tempC);
      ezoMaybeSetTemp(1, EZO_ADDR_EC, tempC);
      ezoMaybeSetTemp(2, EZO_ADDR_DO, tempC);
    }
    float ph    = ezoReadFloat(EZO_ADDR_PH);
    float ec    = ezoReadFloat(EZO_ADDR_EC);
    float doMgL = ezoReadFloat(EZO_ADDR_DO);

    // Live stream
    if (!isnan(tempC)) mavSendNamedFloat("WTR_TEMP", tempC);
    if (!isnan(ph))    mavSendNamedFloat("WTR_PH",   ph);
    if (!isnan(ec))    mavSendNamedFloat("WTR_EC",   ec);
    if (!isnan(doMgL)) mavSendNamedFloat("WTR_DO",   doMgL);

    // Accumulate only during capture
    if (phase == PHASE_CAPTURE) {
      aT.add(tempC); aPh.add(ph); aEc.add(ec); aDo.add(doMgL);
    }
  }

  // Determine flags
  char flags[40] = "";
  if (!isnan(aEc.mean()) && aEc.mean() < 10.0f) strcat(flags, "DRY ");
  if (aT.nan_count)  strcat(flags, "T_NAN ");
  if (aPh.nan_count) strcat(flags, "PH_NAN ");
  if (aEc.nan_count) strcat(flags, "EC_NAN ");
  if (aDo.nan_count) strcat(flags, "DO_NAN ");
  if (aT.n == 0 && aPh.n == 0 && aEc.n == 0 && aDo.n == 0)
    strcpy(flags, "NO_DATA");
  else if (flags[0] == '\0')
    strcpy(flags, "OK");

  // Persist
  sdLogSample(waypointCount, millis(), aT.n,
              aT.mean(), aPh.mean(), aEc.mean(), aDo.mean(),
              aT.stdev(), aPh.stdev(), aEc.stdev(), aDo.stdev(),
              flags);

  // MAVLink averaged sample
  if (!isnan(aT.mean()))  mavSendNamedFloat("WP_TEMP", aT.mean());
  if (!isnan(aPh.mean())) mavSendNamedFloat("WP_PH",   aPh.mean());
  if (!isnan(aEc.mean())) mavSendNamedFloat("WP_EC",   aEc.mean());
  if (!isnan(aDo.mean())) mavSendNamedFloat("WP_DO",   aDo.mean());
  mavSendNamedFloat("WP_NUM", (float)waypointCount);

  snprintf(msg, sizeof(msg), "ATLAS: WP %lu done n=%u %s",
           waypointCount, aT.n, flags);
  mavSendStatusText(msg);

  MavSerial.flush();
}

// ============== setup / loop ==============
void setup() {
  Serial.begin(115200);
  delay(200);
  Serial.println("\n[Atlas] boot");

  // Disable unused radios to reduce idle current.
  WiFi.mode(WIFI_OFF);
  btStop();
  esp_bt_controller_disable();

  Wire.begin(I2C_SDA, I2C_SCL, I2C_FREQ);
  MavSerial.begin(MAV_BAUD, SERIAL_8N1, MAV_RX_PIN, MAV_TX_PIN);
  sdInit();

  mavSendStatusText("ATLAS: payload online");
  Serial.println("[Atlas] ready, listening for EXTENDED_SYS_STATE");
}

void loop() {
  pollMavlink();

  if (currLandedState != lastLandedState) {
    Serial.printf("[Atlas] landed_state: %u -> %u\n",
                  lastLandedState, currLandedState);

    bool wasAirborne = (lastLandedState == MAV_LANDED_STATE_IN_AIR ||
                        lastLandedState == MAV_LANDED_STATE_LANDING ||
                        lastLandedState == MAV_LANDED_STATE_TAKEOFF);
    bool nowLanded = (currLandedState == MAV_LANDED_STATE_ON_GROUND);

    if (wasAirborne && nowLanded) {
      Serial.println("[Atlas] LANDING detected -> running cycle");
      runMeasurementCycle();
    }

    lastLandedState = currLandedState;
  }

  delay(IDLE_POLL_MS);
}
