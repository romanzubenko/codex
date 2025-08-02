// Mini TypeScript agent demonstrating a tiny subset of Codex features.
// Run with: `ts-node mini_agent.ts` (requires Node >= 18 for global fetch).

import * as fs from 'fs/promises';
import { exec as execCb } from 'child_process';
import { promisify } from 'util';

const exec = promisify(execCb);

/** Types for our in-memory queue operations. */
type Op =
  | { type: 'configure'; model: string }
  | { type: 'user'; text: string };

/** Basic event object we log to the console. */
type Event = { type: string; data?: any };

/**
 * Extremely small agent that processes operations from a submission queue.
 * It keeps a list of events for transparency and interacts with the OpenAI
 * API, a shell, and a naive patch applicator. This is intentionally
 * single-threaded and stores queues in memory.
 */
class MiniAgent {
  private submissionQueue: Op[] = [];
  private eventQueue: Event[] = [];
  private model = 'gpt-4o-mini';
  private prompt: string;
  private nextTaskId = 1;
  private plan: string[] = [];

  constructor(prompt: string) {
    this.prompt = prompt;
  }

  /** Push an operation onto the submission queue. */
  submit(op: Op) {
    this.submissionQueue.push(op);
  }

  /** Simple console logger for events. */
  private logEvent(event: Event) {
    this.eventQueue.push(event);
    console.log(`[event] ${event.type}`, event.data ?? '');
  }

  /** Process operations until the queue is empty. */
  async run() {
    while (this.submissionQueue.length > 0) {
      const op = this.submissionQueue.shift()!;
      if (op.type === 'configure') {
        this.model = op.model;
        this.logEvent({ type: 'SessionConfigured', data: { model: this.model } });
      } else if (op.type === 'user') {
        const id = this.nextTaskId++;
        await this.runTask(id, op.text);
      }
    }
  }

  /**
   * Handles a user task: build a conversation with the model and loop until it
   * responds with a final assistant message. Tool outputs are appended to the
   * conversation to form the next turn's input.
   */
  private async runTask(id: number, text: string) {
    this.logEvent({ type: 'TaskStarted', data: { id, text } });

    const messages: any[] = [
      { role: 'system', content: this.prompt },
      { role: 'user', content: text },
    ];

    while (true) {
      const reply = await this.callOpenAI(messages);

      if (reply.tool_calls && reply.tool_calls.length > 0) {
        // Execute each tool sequentially and feed its output back to the model.
        for (const call of reply.tool_calls) {
          await this.handleTool(call, messages, id);
        }
      } else if (reply.content) {
        // No tool calls: task is complete.
        this.logEvent({
          type: 'AssistantMessage',
          data: { id, content: reply.content },
        });
        this.logEvent({ type: 'TaskComplete', data: { id } });
        return;
      } else {
        this.logEvent({ type: 'UnknownModelReply', data: reply });
        return;
      }
    }
  }

  /** Execute a tool call from the model. */
  private async handleTool(call: any, messages: any[], taskId: number) {
    const name = call.function.name;
    const args = JSON.parse(call.function.arguments || '{}');

    switch (name) {
      case 'shell':
        this.logEvent({ type: 'ExecStart', data: { taskId, command: args.command } });
        const { stdout, stderr } = await exec(args.command, { cwd: process.cwd() });
        this.logEvent({ type: 'ExecEnd', data: { stdout, stderr } });
        messages.push({ role: 'tool', tool_call_id: call.id, content: stdout + stderr });
        break;

      case 'apply_patch':
        await applyPatch(args.patch);
        this.logEvent({ type: 'PatchApplied', data: { file: args.patch } });
        messages.push({ role: 'tool', tool_call_id: call.id, content: 'Patch applied' });
        break;

      case 'update_plan':
        this.plan = args.steps || [];
        this.logEvent({ type: 'PlanUpdated', data: this.plan });
        messages.push({ role: 'tool', tool_call_id: call.id, content: 'Plan stored' });
        break;

      default:
        this.logEvent({ type: 'UnknownTool', data: name });
        messages.push({ role: 'tool', tool_call_id: call.id, content: 'Unknown tool' });
    }
  }

  /**
   * Minimal wrapper around OpenAI's chat completions API. If no API key is set
   * the call is skipped and a fake response is returned.
   */
  private async callOpenAI(messages: any[]) {
    const apiKey = process.env.OPENAI_API_KEY;
    if (!apiKey) {
      // For offline experimentation return a mock response that asks to end.
      this.logEvent({ type: 'ModelSkipped', data: 'No OPENAI_API_KEY set' });
      return { content: '(no model call, API key missing)' };
    }

    const body = {
      model: this.model,
      messages,
      tools: toolSchema,
    };

    const res = await fetch('https://api.openai.com/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
      },
      body: JSON.stringify(body),
    });

    const json = await res.json();
    this.logEvent({ type: 'ModelResponse', data: json });
    return json.choices[0].message;
  }
}

/** Definition of tools exposed to the model. */
const toolSchema = [
  {
    type: 'function',
    function: {
      name: 'shell',
      description: 'Run a shell command in the current directory',
      parameters: {
        type: 'object',
        properties: { command: { type: 'string' } },
        required: ['command'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'apply_patch',
      description: 'Apply a simple patch to a file',
      parameters: {
        type: 'object',
        properties: { patch: { type: 'string' } },
        required: ['patch'],
      },
    },
  },
  {
    type: 'function',
    function: {
      name: 'update_plan',
      description: 'Store the current plan for the task',
      parameters: {
        type: 'object',
        properties: {
          steps: { type: 'array', items: { type: 'string' } },
        },
        required: ['steps'],
      },
    },
  },
];

/**
 * Very naive patch applicator. It only supports patches of the form:
 *
 * *** Begin Patch
 * *** Update File: path/to/file
 * @@
 * new file contents
 * @@
 * *** End Patch
 */
async function applyPatch(patch: string) {
  const pathMatch = patch.match(/\*\*\* Update File: (.*)\n/);
  const contentMatch = patch.match(/@@\n([\s\S]*)\n@@/);
  if (!pathMatch || !contentMatch) throw new Error('Invalid patch format');

  const filePath = pathMatch[1].trim();
  const newContent = contentMatch[1];
  await fs.writeFile(filePath, newContent, 'utf8');
  console.log(`[patch] wrote ${filePath}`);
}

// --- Startup sequence -----------------------------------------------------
(async () => {
  const prompt = await fs.readFile('prompt.md', 'utf8');
  const agent = new MiniAgent(prompt);

  // Example operations. Real usage would push to the agent's queue dynamically.
  agent.submit({ type: 'configure', model: 'gpt-4o-mini' });
  agent.submit({
    type: 'user',
    text: 'Fix UI bug on login page, it should check if user logged in first before redirecting.',
  });

  await agent.run();
})();

