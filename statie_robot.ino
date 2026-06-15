#include <WiFi.h>
#include <PubSubClient.h>

const char* ssid = "AlinaiPhone";
const char* password = "tenis2026abc";
const char* mqtt_server = "broker.hivemq.com";

// CONFIGURARE HARDWARE STATIE
const int PIN_LEDURI = 4; // Pinul GPIO4 pentru controlul LED-urilor verzi

WiFiClient espClient;
PubSubClient client(espClient);

// FUNCTIA DE ASCULTARE SI REACTIE
void callback(char* topic, byte* payload, unsigned int length) {
  String mesaj = "";
  for (int i = 0; i < length; i++) {
    mesaj += (char)payload[i];
  }
  
  Serial.print("\n>>> ALERTĂ: Mesaj primit de la Robot: ");
  Serial.println(mesaj);

  if (mesaj == "INTOARCERE") {
    Serial.println(">>> COMANDĂ VALIDATĂ! Pornesc sistemele de ghidare...");
    
    // 1. APRINDEREA FIZICĂ A LED-URILOR VERZI
    digitalWrite(PIN_LEDURI, HIGH);
    Serial.println(">>> [HARDWARE] LED-urile verzi de pe pinul 4 au fost APRINSE!");
    
    // 2. Sincronizare canal și pornire Far Wi-Fi
    int canal_iphone = WiFi.channel();
    Serial.print(">>> iPhone-ul emite pe canalul: ");
    Serial.println(canal_iphone);
    
    WiFi.mode(WIFI_AP_STA); 
    delay(100); 
    
    if(WiFi.softAP("Far_Statie_Baza", "12345678", canal_iphone)) {
      Serial.println(">>> FARUL ESTE ACTIV SI VIZIBIL! Robotul poate veni.");
    } else {
      Serial.println(">>> EROARE: Cipul a refuzat crearea retelei!");
    }
  }
  
  // Opțional: Dacă robotul trimite "SISTEM_PORNIT" sau "ANDOCAT", putem stinge LED-urile
  if (mesaj == "SISTEM_PORNIT") {
    digitalWrite(PIN_LEDURI, LOW);
    Serial.println(">>> [HARDWARE] Resetare: LED-urile au fost STINSE.");
  }
}

void reconnect() {
  while (!client.connected()) {
    Serial.print("Reconectare la MQTT...");
    if (client.connect("StatieBaza_Licenta_12345")) {
      Serial.println("SUCCES!");
      client.subscribe("licenta_robot/status");
    } else {
      Serial.print("Esuat, reincerc in 3 secunde...");
      delay(3000);
    }
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  // CONFIGURARE PIN HARDWARE
  pinMode(PIN_LEDURI, OUTPUT);
  digitalWrite(PIN_LEDURI, LOW); // Ne asigurăm că pornesc STINSE la boot

  // CONFIGURARE WIFI PENTRU IPHONE
  WiFi.disconnect(true, true);
  delay(500);
  WiFi.mode(WIFI_STA);
  WiFi.setTxPower(WIFI_POWER_8_5dBm);
  
  WiFi.begin(ssid, password);
  
  Serial.print("\nConectare la iPhone...");
  int timeout = 0;
  while (WiFi.status() != WL_CONNECTED && timeout < 30) {
    delay(500);
    Serial.print(".");
    timeout++;
  }
  
  if (WiFi.status() == WL_CONNECTED) {
    Serial.println("\nConectat la WiFi! IP: " + WiFi.localIP().toString());
    
    client.setServer(mqtt_server, 1883);
    client.setCallback(callback); 
    
    Serial.print("Conectare la serverul MQTT...");
    if (client.connect("StatieBaza_Licenta_12345")) {
      Serial.println("SUCCES!");
      client.subscribe("licenta_robot/status"); 
      Serial.println("Sistemul este armat si asculta comenzile.");
    } else {
      Serial.println("Esec la MQTT!");
    }
  } else {
    Serial.println("\nEsuat! Status: " + String(WiFi.status()));
  }
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    if (!client.connected()) {
      reconnect(); 
    }
    client.loop(); 
  }
}