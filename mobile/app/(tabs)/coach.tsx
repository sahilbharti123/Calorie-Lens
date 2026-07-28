import { useRouter } from 'expo-router';
import { useMemo, useRef, useState } from 'react';
import {
  ActivityIndicator,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';
import { SafeAreaView } from 'react-native-safe-area-context';

import { Glyph } from '@/src/components/glyph';
import { apiRequest } from '@/src/lib/api-client';
import { dayTotals } from '@/src/lib/stats';
import { useApp } from '@/src/store/app-store';
import { useAuth } from '@/src/store/auth-store';
import { palette, radius, space, type } from '@/src/theme';
import type { CoachMessage } from '@/src/types';

function offlineReply(message: string, calories: number, protein: number, target: number) {
  const remaining = Math.max(0, target - calories);
  const lowered = message.toLowerCase();
  if (/workout|gym|train|exercise/.test(lowered)) {
    return 'A simple session works: 5–10 minutes easy warm-up, your planned strength or cardio work, then a short cool-down. Keep pain—not normal effort—as a stop signal.';
  }
  if (/dinner|eat|meal|protein|food/.test(lowered)) {
    return `You have about ${Math.round(remaining)} kcal left today and have logged ${Math.round(protein)} g protein. Build the next meal around a protein you enjoy, then add vegetables and a measured carb portion.`;
  }
  if (/water|hydr/.test(lowered)) {
    return 'Sip steadily rather than catching up all at once. Use the water tile on Today to log each 250 ml.';
  }
  return `Today you have logged about ${Math.round(calories)} kcal and ${Math.round(protein)} g protein. Pick one small next action: log your next meal, drink 250 ml water, or take a 10-minute walk.`;
}

export default function CoachScreen() {
  const router = useRouter();
  const { data, today, addCoachMessage } = useApp();
  const { session } = useAuth();
  const [text, setText] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const scrollRef = useRef<ScrollView>(null);
  const totals = useMemo(() => dayTotals(today), [today]);
  const messages = data.coachMessages;

  async function send() {
    const value = text.trim();
    if (!value || busy) return;
    setText('');
    setBusy(true);
    setError('');
    const userMessage = addCoachMessage({ role: 'user', text: value });
    try {
      let reply: string;
      if (session) {
        const recent = [...messages, userMessage].slice(-12).map(({ role, text: body }) => ({
          role,
          text: body,
        }));
        const response = await apiRequest<{ reply: string }>('/v1/coach', {
          method: 'POST',
          body: JSON.stringify({
            message: value,
            today: { ...today, goals: data.goals, totals },
            memory: data.coachMemory,
            recent_messages: recent,
          }),
        }, session);
        reply = response.reply;
      } else {
        reply = offlineReply(value, totals.calories, totals.protein, data.goals.calories);
      }
      addCoachMessage({ role: 'coach', text: reply });
    } catch (reason) {
      const fallback = offlineReply(value, totals.calories, totals.protein, data.goals.calories);
      addCoachMessage({ role: 'coach', text: fallback });
      setError(`${reason instanceof Error ? reason.message : 'Coach service is unavailable.'} Showing an offline suggestion.`);
    } finally {
      setBusy(false);
      requestAnimationFrame(() => scrollRef.current?.scrollToEnd({ animated: true }));
    }
  }

  return (
    <SafeAreaView style={styles.safe} edges={['top']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 84 : 0}
        style={{ flex: 1 }}>
        <View style={styles.header}>
          <View>
            <Text style={styles.eyebrow}>YOUR FITNESS EXPERT</Text>
            <Text style={styles.title}>Coach</Text>
          </View>
          <Pressable onPress={() => router.push('/quick-log')} style={styles.logButton}>
            <Glyph name="mic" color={palette.lime} size={19} />
            <Text style={styles.logText}>Log</Text>
          </Pressable>
        </View>

        <View style={styles.context}>
          <View>
            <Text style={styles.contextNumber}>{Math.round(totals.calories)}</Text>
            <Text style={styles.contextLabel}>KCAL LOGGED</Text>
          </View>
          <View style={styles.contextDivider} />
          <View>
            <Text style={styles.contextNumber}>{Math.round(totals.protein)} g</Text>
            <Text style={styles.contextLabel}>PROTEIN</Text>
          </View>
          <View style={styles.contextDivider} />
          <View>
            <Text style={styles.contextNumber}>{today.steps.toLocaleString()}</Text>
            <Text style={styles.contextLabel}>STEPS</Text>
          </View>
        </View>

        <ScrollView
          ref={scrollRef}
          contentContainerStyle={styles.messages}
          keyboardShouldPersistTaps="handled"
          onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}>
          {messages.length === 0 ? (
            <View style={styles.welcome}>
              <View style={styles.welcomeIcon}><Glyph name="spark" color={palette.forest} size={24} /></View>
              <Text style={styles.welcomeTitle}>Ask about your day.</Text>
              <Text style={styles.welcomeBody}>
                I can use your logged meals, activity, goals and saved preferences to suggest a realistic next step.
              </Text>
              <View style={styles.prompts}>
                {['What should I eat for dinner?', 'Plan a quick workout', 'How is my day going?'].map((prompt) => (
                  <Pressable key={prompt} onPress={() => setText(prompt)} style={styles.prompt}>
                    <Text style={styles.promptText}>{prompt}</Text>
                  </Pressable>
                ))}
              </View>
            </View>
          ) : messages.map((message: CoachMessage) => (
            <View
              key={message.id}
              style={[
                styles.bubble,
                message.role === 'user' ? styles.userBubble : styles.coachBubble,
              ]}>
              {message.role === 'coach' ? <Text style={styles.bubbleLabel}>COACH</Text> : null}
              <Text style={[styles.bubbleText, message.role === 'user' && styles.userText]}>
                {message.text}
              </Text>
            </View>
          ))}
          {busy ? (
            <View style={[styles.bubble, styles.coachBubble, styles.thinking]}>
              <ActivityIndicator color={palette.forest} size="small" />
              <Text style={styles.thinkingText}>Thinking through today’s data…</Text>
            </View>
          ) : null}
          {error ? <Text style={styles.error}>{error}</Text> : null}
        </ScrollView>

        <View style={styles.composer}>
          <TextInput
            multiline
            onChangeText={setText}
            onSubmitEditing={() => void send()}
            placeholder="Ask your coach…"
            placeholderTextColor="#8A938B"
            returnKeyType="send"
            style={styles.input}
            value={text}
          />
          <Pressable disabled={!text.trim() || busy} onPress={() => void send()} style={[styles.send, (!text.trim() || busy) && styles.sendDisabled]}>
            <Glyph name="chevron" color={palette.lime} size={18} />
          </Pressable>
        </View>
      </KeyboardAvoidingView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  safe: { flex: 1, backgroundColor: palette.canvas },
  header: { paddingHorizontal: space.md, paddingTop: 8, paddingBottom: 15, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  eyebrow: { color: palette.muted, fontFamily: type.demi, fontSize: 9.5, letterSpacing: 1.4, marginBottom: 3 },
  title: { color: palette.ink, fontFamily: type.demi, fontSize: 30, letterSpacing: -1 },
  logButton: { height: 42, flexDirection: 'row', alignItems: 'center', gap: 7, paddingHorizontal: 14, backgroundColor: palette.forest, borderRadius: radius.pill },
  logText: { color: palette.lime, fontFamily: type.demi, fontSize: 11 },
  context: { marginHorizontal: space.md, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderRadius: radius.md, padding: 13, flexDirection: 'row', alignItems: 'center', justifyContent: 'space-around' },
  contextNumber: { color: palette.ink, fontFamily: type.demi, fontSize: 15, textAlign: 'center' },
  contextLabel: { color: palette.muted, fontFamily: type.demi, fontSize: 7.5, letterSpacing: 0.8, marginTop: 2, textAlign: 'center' },
  contextDivider: { width: 1, height: 28, backgroundColor: palette.line },
  messages: { flexGrow: 1, padding: space.md, paddingBottom: 18 },
  welcome: { alignItems: 'center', paddingTop: 36, paddingHorizontal: 22 },
  welcomeIcon: { width: 54, height: 54, borderRadius: 18, backgroundColor: palette.lime, alignItems: 'center', justifyContent: 'center' },
  welcomeTitle: { color: palette.ink, fontFamily: type.demi, fontSize: 21, letterSpacing: -0.5, marginTop: 14 },
  welcomeBody: { color: palette.muted, fontFamily: type.regular, fontSize: 12, lineHeight: 18, textAlign: 'center', marginTop: 6 },
  prompts: { alignSelf: 'stretch', marginTop: 20, gap: 8 },
  prompt: { minHeight: 44, borderRadius: radius.pill, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, paddingHorizontal: 16, alignItems: 'center', justifyContent: 'center' },
  promptText: { color: palette.ink, fontFamily: type.medium, fontSize: 11 },
  bubble: { maxWidth: '86%', padding: 13, borderRadius: 18, marginBottom: 9 },
  userBubble: { alignSelf: 'flex-end', backgroundColor: palette.forest, borderBottomRightRadius: 5 },
  coachBubble: { alignSelf: 'flex-start', backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, borderBottomLeftRadius: 5 },
  bubbleLabel: { color: palette.limeDark, fontFamily: type.demi, fontSize: 8, letterSpacing: 1, marginBottom: 5 },
  bubbleText: { color: palette.ink, fontFamily: type.regular, fontSize: 12.5, lineHeight: 19 },
  userText: { color: palette.white },
  thinking: { flexDirection: 'row', alignItems: 'center', gap: 9 },
  thinkingText: { color: palette.muted, fontFamily: type.regular, fontSize: 10.5 },
  error: { color: palette.coral, fontFamily: type.regular, fontSize: 9.5, lineHeight: 14, marginTop: 4 },
  composer: { flexDirection: 'row', alignItems: 'flex-end', gap: 9, paddingHorizontal: space.md, paddingTop: 10, paddingBottom: 10, borderTopWidth: 1, borderTopColor: palette.line, backgroundColor: palette.canvas },
  input: { flex: 1, maxHeight: 110, minHeight: 48, borderRadius: 17, backgroundColor: palette.paper, borderWidth: 1, borderColor: palette.line, paddingHorizontal: 14, paddingVertical: 12, color: palette.ink, fontFamily: type.regular, fontSize: 13 },
  send: { width: 48, height: 48, borderRadius: 17, backgroundColor: palette.forest, alignItems: 'center', justifyContent: 'center' },
  sendDisabled: { opacity: 0.38 },
});
