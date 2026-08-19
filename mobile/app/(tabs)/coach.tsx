import { useMemo, useRef, useState } from 'react';
import {
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from 'react-native';

import { Glyph } from '@/src/components/glyph';
import { Card, PrimaryButton, Screen, ScreenHeader, Tap } from '@/src/components/ui';
import { personalDailyNudge, personalOfflineReply } from '@/src/lib/personalization';
import { useApp } from '@/src/store/app-store';
import { alpha, palette, radius, space, text } from '@/src/theme';

const PROMPTS = ['What should I eat?', 'Plan my workout', 'How am I doing?'] as const;

export default function CoachScreen() {
  const { addCoachMessage, data, today } = useApp();
  const [draft, setDraft] = useState('');
  const scrollRef = useRef<ScrollView>(null);
  const nudge = useMemo(() => personalDailyNudge(data, today), [data, today]);

  function send(value = draft) {
    const question = value.trim();
    if (!question) return;
    addCoachMessage({ role: 'user', text: question });
    addCoachMessage({ role: 'coach', text: personalOfflineReply(question, data, today) });
    setDraft('');
    Keyboard.dismiss();
    requestAnimationFrame(() => scrollRef.current?.scrollToEnd({ animated: true }));
  }

  return (
    <Screen edges={['bottom']}>
      <KeyboardAvoidingView
        behavior={Platform.OS === 'ios' ? 'padding' : undefined}
        keyboardVerticalOffset={8}
        style={styles.flex}>
        <ScrollView
          contentContainerStyle={styles.content}
          keyboardDismissMode="on-drag"
          keyboardShouldPersistTaps="handled"
          onScrollBeginDrag={Keyboard.dismiss}
          onTouchMove={Keyboard.dismiss}
          ref={scrollRef}
          showsVerticalScrollIndicator={false}>
          <ScreenHeader eyebrow="Private · works offline" title="Coach" />

          <Card glow raised style={styles.nudgeCard}>
            <View style={styles.nudgeIcon}>
              <Glyph color={palette.onLime} name="spark" size={21} />
            </View>
            <View style={styles.nudgeCopy}>
              <Text style={styles.nudgeTitle}>{nudge.title}</Text>
              <Text style={styles.nudgeBody}>{nudge.body}</Text>
            </View>
          </Card>

          <ScrollView
            contentContainerStyle={styles.prompts}
            horizontal
            showsHorizontalScrollIndicator={false}>
            {PROMPTS.map((prompt) => (
              <Tap
                accessibilityLabel={`Ask coach: ${prompt}`}
                key={prompt}
                onPress={() => send(prompt)}
                style={styles.prompt}>
                <Text style={styles.promptText}>{prompt}</Text>
              </Tap>
            ))}
          </ScrollView>

          <View accessibilityLiveRegion="polite" style={styles.messages}>
            {data.coachMessages.length ? data.coachMessages.map((message) => (
              <View
                key={message.id}
                style={[
                  styles.bubble,
                  message.role === 'user' ? styles.userBubble : styles.coachBubble,
                ]}>
                {message.role === 'coach' ? (
                  <Glyph color={palette.lime} name="spark" size={15} />
                ) : null}
                <Text style={message.role === 'user' ? styles.userText : styles.coachText}>
                  {message.text}
                </Text>
              </View>
            )) : (
              <View style={styles.empty}>
                <Text style={styles.emptyTitle}>Ask about today</Text>
                <Text style={styles.emptyBody}>
                  Answers use only your goals and logs. They are general guidance, not medical advice.
                </Text>
              </View>
            )}
          </View>
        </ScrollView>

        <View style={styles.composer}>
          <TextInput
            accessibilityLabel="Message your coach"
            blurOnSubmit
            multiline
            onChangeText={setDraft}
            onSubmitEditing={() => send()}
            placeholder="Ask about food, training or your target…"
            placeholderTextColor={palette.inkLow}
            returnKeyType="send"
            selectionColor={palette.lime}
            style={styles.input}
            value={draft}
          />
          <PrimaryButton
            compact
            disabled={!draft.trim()}
            icon="arrowUp"
            label="Send"
            onPress={() => send()}
          />
        </View>
      </KeyboardAvoidingView>
    </Screen>
  );
}

const styles = StyleSheet.create({
  flex: { flex: 1 },
  content: { paddingHorizontal: space.md, paddingBottom: 132 },
  nudgeCard: { marginBottom: space.md },
  nudgeIcon: {
    width: 42,
    height: 42,
    borderRadius: 15,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: palette.lime,
  },
  nudgeCopy: { marginTop: space.sm, gap: 4 },
  nudgeTitle: { ...text.title, fontSize: 21, color: palette.ink },
  nudgeBody: { ...text.body, color: palette.inkMid },
  prompts: { gap: 8, paddingBottom: space.md },
  prompt: {
    minHeight: 42,
    justifyContent: 'center',
    paddingHorizontal: 14,
    borderRadius: radius.pill,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: alpha.limeFaint,
  },
  promptText: { ...text.value, fontSize: 13, color: palette.lime },
  messages: { gap: 10 },
  bubble: {
    maxWidth: '88%',
    flexDirection: 'row',
    alignItems: 'flex-start',
    gap: 8,
    paddingHorizontal: 14,
    paddingVertical: 12,
    borderRadius: radius.md,
  },
  userBubble: { alignSelf: 'flex-end', backgroundColor: palette.lime },
  coachBubble: {
    alignSelf: 'flex-start',
    backgroundColor: palette.surface,
    borderWidth: 1,
    borderColor: palette.line,
  },
  userText: { ...text.body, flexShrink: 1, color: palette.onLime },
  coachText: { ...text.body, flexShrink: 1, color: palette.ink },
  empty: {
    alignItems: 'center',
    gap: 5,
    paddingVertical: 32,
    paddingHorizontal: 22,
  },
  emptyTitle: { ...text.title, fontSize: 18, color: palette.ink },
  emptyBody: { ...text.body, color: palette.inkLow, textAlign: 'center' },
  composer: {
    position: 'absolute',
    left: 0,
    right: 0,
    bottom: 0,
    flexDirection: 'row',
    alignItems: 'flex-end',
    gap: 9,
    paddingHorizontal: space.md,
    paddingTop: 10,
    paddingBottom: 10,
    borderTopWidth: 1,
    borderTopColor: palette.line,
    backgroundColor: palette.bg,
  },
  input: {
    ...text.body,
    flex: 1,
    minHeight: 48,
    maxHeight: 110,
    paddingHorizontal: 14,
    paddingVertical: 12,
    color: palette.ink,
    borderRadius: radius.md,
    borderWidth: 1,
    borderColor: palette.lineHi,
    backgroundColor: palette.surface,
  },
});
